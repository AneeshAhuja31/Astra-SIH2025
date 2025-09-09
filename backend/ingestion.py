import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import date 
import json
load_dotenv()

USER = os.getenv("SUPABASE_USER")
PASSWORD = os.getenv("SUPABASE_PASSWORD")
HOST = os.getenv("SUPABASE_HOST")
PORT = os.getenv("SUPABASE_PORT")
DBNAME = os.getenv("SUPABASE_DBNAME")

folder_path = r"C:\Users\HP\OneDrive\Desktop\sih_proj\Astra-SIH2025\cut_output_true"

def insert_in_db():
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        print("Connection successful!")
        
        cursor = connection.cursor()
        
        for year in range(2001, 2013):
            file_path = rf"{folder_path}\mapped_argo_details_{year}_sorted.csv"
            dataframe = pd.read_csv(file_path)
            try:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS argo_data_{year}(
                        ad_observation_id VARCHAR(255),
                        depth NUMERIC,
                        temperature NUMERIC,
                        density NUMERIC,
                        salinity NUMERIC,
                        ao_observation_id VARCHAR(255),
                        latitude NUMERIC,
                        longitude NUMERIC,
                        date DATE,
                        region VARCHAR(255)
                    )
                    """
                )
                connection.commit()
                print(f"Table argo_data_{year} created successfully.")
            except Exception as e:
                print(f"Failed to create table for year {year}: {e}")
                continue
            
            # for _, row in dataframe.iterrows():
            #     try:
            #         # date_ = row["date"].split('/')
            #         # date_ = date(date_[0],date_[1],date_[2])
            #         cursor.execute(
            #             f"""
            #             INSERT INTO argo_data_{year} (ad_observation_id, depth, temperature, density, salinity, latitude, longitude, date, region)
            #             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            #             """,
            #             (
            #                 row['ad_observation_id'],
            #                 row['depth'],
            #                 row['temperature'],
            #                 row['density'],
            #                 row['salinity'],
            #                 row['latitude'],
            #                 row['longitude'],
            #                 row['date'],
            #                 row['region']
            #             )
            #         )
            #         connection.commit()
            #     except Exception as e:
            #         print(f"Failed to insert data for year {year}, row {row['ad_observation_id']}: {e}")
            #         connection.rollback()  # Rollback the transaction for this row
            #     else:
            #         connection.commit()  # Commit the transaction for successful insertion
            
            # print(f"Data inserted for year {year}.")
        
        cursor.close()
        connection.close()
        print("Connection closed.")

    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    insert_in_db()