import pandas as pd
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --------------------
# 1. Region Mapping
# --------------------
def get_region(lat, lon):
    """
    Returns the ocean region based on simple bounding boxes.
    These are just examples – can be refined with real shapefiles later.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"

    if 5 <= lat <= 22 and 80 <= lon <= 100:
        return "Bay of Bengal"
    elif 10 <= lat <= 25 and 60 <= lon <= 75:
        return "Arabian Sea"
    elif -10 <= lat <= 10 and 180 >= lon >= -180:
        return "Equatorial Region"
    else:
        return "Indian Ocean"

# --------------------
# 2. Load Data
# --------------------
df = pd.read_csv("sample_data.csv")

# Extract date from ad_observation_id
df["date"] = df["ad_observation_id"].apply(lambda x: x.split("_")[1])
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

df["month"] = df["date"].dt.month_name()
df["year"] = df["date"].dt.year

# Round depth for grouping
df["depth_level"] = df["depth"].round(-1).astype(int)

# Add region column if lat/lon present, else Unknown
if "lat" in df.columns and "lon" in df.columns:
    df["region"] = df.apply(lambda row: get_region(row["lat"], row["lon"]), axis=1)
else:
    df["region"] = "Unknown"

# --------------------
# 3. Initialize LLM
# --------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an oceanography assistant. 
Summarize the following aggregated ocean data in one line, highlighting any notable temperature or salinity trends.

Region: {region}
Month: {month}
Year: {year}
Depth: {depth_level} m
Average Temperature: {temperature_avg:.3f} °C
Average Salinity: {salinity_avg:.3f} PSU
Sample Count: {sample_count}
""")

# --------------------
# 4. Aggregate + Summarize
# --------------------
json_records = []

for (region, month, year, depth), group in df.groupby(["region", "month", "year", "depth_level"]):
    temperature_avg = group["temperature"].mean()
    salinity_avg = group["salinity"].mean()
    sample_count = len(group)

    # Generate note using LLM
    response = llm.invoke(prompt.format(
        region=region,
        month=month,
        year=year,
        depth_level=depth,
        temperature_avg=temperature_avg,
        salinity_avg=salinity_avg,
        sample_count=sample_count
    ))
    note = response.content.strip()

    record = {
        "region": region,
        "month": month,
        "year": int(year),
        "depth_level": int(depth),
        "temperature_avg": round(temperature_avg, 3),
        "salinity_avg": round(salinity_avg, 3),
        "sample_count": sample_count,
        "note": note
    }

    json_records.append(record)

# --------------------
# 5. Save JSON
# --------------------
with open("argo_summary.json", "w") as f:
    json.dump(json_records, f, indent=2)

print("✅ JSON file created: argo_summary.json")
