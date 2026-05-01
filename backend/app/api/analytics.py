from sqlalchemy import text
from fastapi import APIRouter

from app.db.database import DbSession

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"]
)

@router.get("/monthly-trends")
async def monthly_trends(db: DbSession):

    query = text("""
        SELECT
            DATE_TRUNC('month', published_at) AS month,
            sentiment,
            COUNT(*) AS count
        FROM comments
        GROUP BY month, sentiment
        ORDER BY month ASC
    """)

    result = await db.execute(query)
    rows = result.fetchall()

    data_map = {}

    for r in rows:
        month = r.month.strftime("%Y-%m")

        if month not in data_map:
            data_map[month] = {
                "complaint": 0,
                "question": 0,
                "praise": 0,
                "suggestion": 0,
                "statement": 0
            }

        # sentiment mapping
        if r.sentiment == "negative":
            data_map[month]["complaint"] += r.count

        elif r.sentiment == "positive":
            data_map[month]["praise"] += r.count

        elif r.sentiment == "neutral":
            data_map[month]["statement"] += r.count

        else:
            data_map[month]["statement"] += r.count

    months_sorted = sorted(data_map.keys())

    return {
        "months": months_sorted,
        "complaints": [data_map[m]["complaint"] for m in months_sorted],
        "questions": [data_map[m]["question"] for m in months_sorted],
        "praise": [data_map[m]["praise"] for m in months_sorted],
        "suggestions": [data_map[m]["suggestion"] for m in months_sorted],
        "statements": [data_map[m]["statement"] for m in months_sorted],
    }