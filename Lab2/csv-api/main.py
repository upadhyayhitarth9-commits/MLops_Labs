from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import os
import io
from typing import Optional

app = FastAPI(
    title="CSV Data API",
    description="Upload, analyze, and query CSV files",
    version="1.0.0"
)

# Directory to store uploaded CSVs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    """Welcome endpoint"""
    return {
        "message": "CSV Data API",
        "endpoints": {
            "POST /upload": "Upload a CSV file",
            "GET /files": "List all uploaded files",
            "GET /files/{filename}/info": "Get file summary",
            "GET /files/{filename}/data": "Get file data with optional filtering",
            "DELETE /files/{filename}": "Delete a file"
        }
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    # Validate it's a readable CSV
    try:
        df = pd.read_csv(filepath)
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")


@app.get("/files")
def list_files():
    """List all uploaded CSV files"""
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
    return {"files": files, "count": len(files)}


@app.get("/files/{filename}/info")
def get_file_info(filename: str):
    """Get summary statistics for a CSV file"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    df = pd.read_csv(filepath)
    
    # Build column info
    columns_info = {}
    for col in df.columns:
        columns_info[col] = {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null": int(df[col].isna().sum()),
            "unique": int(df[col].nunique())
        }
        # Add numeric stats if applicable
        if df[col].dtype in ['int64', 'float64']:
            columns_info[col].update({
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None
            })
    
    return {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_details": columns_info,
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    }


@app.get("/files/{filename}/data")
def get_file_data(
    filename: str,
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    sort_by: Optional[str] = None,
    ascending: bool = True,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None
):
    """Get data from a CSV file with optional filtering and pagination"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    df = pd.read_csv(filepath)
    
    # Apply filter if provided
    if filter_column and filter_value:
        if filter_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{filter_column}' not found")
        df = df[df[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]
    
    # Apply sorting
    if sort_by:
        if sort_by not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{sort_by}' not found")
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    total = len(df)
    df = df.iloc[offset:offset + limit]
    
    return {
        "filename": filename,
        "total_rows": total,
        "returned_rows": len(df),
        "offset": offset,
        "data": df.to_dict(orient="records")
    }


@app.get("/files/{filename}/download")
def download_file(filename: str):
    """Download a CSV file"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    df = pd.read_csv(filepath)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Delete a CSV file"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    os.remove(filepath)
    return {"message": f"File '{filename}' deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)