import asyncio
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from typing import List, Dict

from routes.search import perform_search, SearchRequest
from utils import csv_handler

router = APIRouter()

# Load concurrency limit from env
BULK_CONCURRENCY = int(os.getenv("BULK_CONCURRENCY", "5"))

@router.post("/bulk-search")
async def bulk_search(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    contents = await file.read()
    try:
        rows = csv_handler.parse_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no valid rows.")

    semaphore = asyncio.Semaphore(BULK_CONCURRENCY)
    
    async def process_row(row: Dict):
        async with semaphore:
            try:
                res = await perform_search(SearchRequest(
                    company=row["company"],
                    designation=row["designation"]
                ))
                return {
                    "company": row["company"],
                    "designation": row["designation"],
                    "name": res.name,
                    "source": res.source,
                    "confidence": res.confidence,
                    "status": res.status
                }
            except Exception as e:
                return {
                    "company": row["company"],
                    "designation": row["designation"],
                    "name": "Error",
                    "source": "N/A",
                    "confidence": 0,
                    "status": f"Processing failed: {str(e)}"
                }

    tasks = [process_row(row) for row in rows]
    results = await asyncio.gather(*tasks)
    
    csv_bytes = csv_handler.results_to_csv(results)
    
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=results_{file.filename}"
        }
    )
