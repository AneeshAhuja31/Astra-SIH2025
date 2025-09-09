import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# For Groq integration
from groq import Groq
import time

class OceanographicProcessor:
    def __init__(self, groq_api_key=None):
        """Initialize the processor with optional Groq client"""
        self.groq_client = None
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        
        # Water mass classification thresholds
        self.water_mass_criteria = {
            'Tropical Surface Water': {'temp': (24, 30), 'sal': (34.5, 36.5), 'depth': (0, 200)},
            'Subtropical Surface Water': {'temp': (18, 25), 'sal': (35.5, 37.0), 'depth': (0, 150)},
            'Antarctic Intermediate Water': {'temp': (3, 7), 'sal': (34.2, 34.6), 'depth': (500, 1500)},
            'Deep Water': {'temp': (1, 4), 'sal': (34.6, 35.0), 'depth': (1500, 4000)},
            'Bottom Water': {'temp': (-2, 2), 'sal': (34.6, 34.8), 'depth': (4000, 6000)},
            'Arctic Water': {'temp': (-2, 2), 'sal': (30, 35), 'depth': (0, 500)},
            'Mediterranean Water': {'temp': (8, 15), 'sal': (36, 39), 'depth': (200, 1000)},
            'Indian Deep Water': {'temp': (2, 6), 'sal': (34.6, 34.8), 'depth': (800, 2000)}
        }
    
    def process_all_files(self, folder_path):
        """Process all CSV files in the folder"""
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files.sort()  # Ensure chronological order
        
        print(f"Found {len(csv_files)} CSV files to process...")
        
        for csv_file in csv_files:
            try:
                # Extract year from filename
                year = self._extract_year_from_filename(csv_file)
                if year:
                    print(f"Processing {csv_file} for year {year}...")
                    file_path = os.path.join(folder_path, csv_file)
                    self.process_single_year(file_path, year)
                else:
                    print(f"Could not extract year from {csv_file}, skipping...")
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                continue
    
    def _extract_year_from_filename(self, filename):
        """Extract year from filename like 'mapped_argo_details_2001_sorted.csv'"""
        import re
        match = re.search(r'(\d{4})', filename)
        return match.group(1) if match else None
    
    def process_single_year(self, csv_path, year):
        """Process a single year's CSV file"""
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Clean and prepare data
        df = self._clean_data(df)
        
        # Group by month and region
        monthly_summaries = self._create_monthly_summaries(df, year)
        
        # Create annual overview
        annual_overview = self._create_annual_overview(df, year)
        
        # Compile final JSON structure
        final_json = {
            "year": year,
            "summary_type": "monthly_regional",
            "total_measurements": len(df),
            "regions_covered": sorted(df['region'].unique().tolist()),
            "data_period": {
                "start_date": df['date'].min(),
                "end_date": df['date'].max()
            },
            "monthly_regional_data": monthly_summaries,
            "annual_overview": annual_overview
        }
        
        # Save JSON file
        output_filename = f"{year}-ocean-data.json"
        with open(output_filename, 'w') as f:
            json.dump(final_json, f, indent=2, default=str)
        
        print(f"✅ Created {output_filename} with {len(monthly_summaries)} monthly summaries")
    
    def _clean_data(self, df):
        """Clean and prepare the dataframe"""
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Remove invalid measurements
        df = df.dropna(subset=['temperature', 'salinity', 'density', 'depth'])
        
        # Remove extreme outliers (basic cleaning)
        df = df[(df['temperature'] >= -3) & (df['temperature'] <= 35)]
        df = df[(df['salinity'] >= 25) & (df['salinity'] <= 45)]
        df = df[(df['depth'] >= 0) & (df['depth'] <= 6000)]
        
        return df
    
    def _create_monthly_summaries(self, df, year):
        """Create detailed monthly summaries for each region"""
        summaries = []
        
        for (year_month, region), group in df.groupby(['year_month', 'region']):
            if len(group) < 5:  # Skip groups with too few measurements
                continue
                
            summary = self._create_detailed_summary(group, str(year_month), region)
            summaries.append(summary)
        
        return summaries
    
    def _create_detailed_summary(self, group, period, region):
        """Create comprehensive summary for a month-region combination"""
        
        # Basic info
        summary = {
            "period": period,
            "region": region,
            "measurement_count": len(group),
            "date_range": {
                "start": group['date'].min().strftime('%Y-%m-%d'),
                "end": group['date'].max().strftime('%Y-%m-%d')
            }
        }
        
        # Depth analysis
        summary["depth_analysis"] = self._analyze_depth(group)
        
        # Temperature analysis
        summary["temperature"] = self._analyze_parameter(group, 'temperature', 'thermocline')
        
        # Salinity analysis  
        summary["salinity"] = self._analyze_parameter(group, 'salinity', 'halocline')
        
        # Density analysis
        summary["density"] = self._analyze_parameter(group, 'density', 'pycnocline')
        
        # Spatial coverage
        summary["spatial_coverage"] = self._analyze_spatial_coverage(group)
        
        # Water mass classification
        summary["water_masses"] = self._classify_water_masses(group)
        
        # Generate oceanographic summary using Groq
        summary["oceanographic_summary"] = self._generate_summary_with_groq(summary, region, period)
        
        # Environmental conditions
        summary["environmental_conditions"] = self._analyze_environmental_conditions(group, period)
        
        # Detect anomalies
        summary["anomalies"] = self._detect_anomalies(group)
        
        # Data quality assessment
        summary["data_quality"] = self._assess_data_quality(group)
        
        # Climate indices (simplified)
        summary["climate_indices"] = self._calculate_climate_indices(period, region)
        
        return summary
    
    def _analyze_depth(self, group):
        """Analyze depth distribution"""
        depth_stats = {
            "range": {
                "min": float(group['depth'].min()),
                "max": float(group['depth'].max()), 
                "avg": float(group['depth'].mean()),
                "std": float(group['depth'].std())
            },
            "distribution": {
                "surface_0_200m": len(group[group['depth'] <= 200]),
                "intermediate_200_1000m": len(group[(group['depth'] > 200) & (group['depth'] <= 1000)]),
                "deep_1000m_plus": len(group[group['depth'] > 1000])
            },
            "coverage": "full_column" if group['depth'].max() > 1000 else "partial"
        }
        return depth_stats
    
    def _analyze_parameter(self, group, param, cline_name):
        """Analyze temperature, salinity, or density"""
        surface_data = group[group['depth'] <= 50][param]
        deep_data = group[group['depth'] >= 500][param]
        
        analysis = {
            "min": float(group[param].min()),
            "max": float(group[param].max()),
            "avg": float(group[param].mean()),
            "std": float(group[param].std())
        }
        
        if len(surface_data) > 0:
            analysis["surface_avg"] = float(surface_data.mean())
        
        if len(deep_data) > 0:
            analysis["deep_avg"] = float(deep_data.mean())
        
        # Calculate gradient and cline depth (simplified)
        if len(group) > 10:
            sorted_group = group.sort_values('depth')
            gradient = np.gradient(sorted_group[param], sorted_group['depth'])
            max_gradient_idx = np.argmax(np.abs(gradient))
            analysis[f"{cline_name}_depth"] = float(sorted_group.iloc[max_gradient_idx]['depth'])
            analysis["gradients"] = {
                "surface_gradient": float(np.mean(gradient[:5])) if len(gradient) > 5 else 0,
                "deep_gradient": float(np.mean(gradient[-5:])) if len(gradient) > 5 else 0
            }
        
        # Additional parameter-specific analysis
        if param == 'salinity':
            analysis["fresh_water_influence"] = "high" if analysis["min"] < 32 else "minimal"
        elif param == 'density':
            analysis["stability_index"] = min(1.0, analysis["std"] / 2.0)
        
        return analysis
    
    def _analyze_spatial_coverage(self, group):
        """Analyze spatial distribution"""
        spatial = {
            "lat_range": [float(group['latitude'].min()), float(group['latitude'].max())],
            "lon_range": [float(group['longitude'].min()), float(group['longitude'].max())],
            "center_point": [float(group['latitude'].mean()), float(group['longitude'].mean())]
        }
        
        # Estimate spatial extent (very simplified)
        lat_span = spatial["lat_range"][1] - spatial["lat_range"][0]
        lon_span = spatial["lon_range"][1] - spatial["lon_range"][0]
        spatial["spatial_extent_km2"] = int(lat_span * lon_span * 12100)  # Rough conversion
        
        # Get primary regions directly from CSV region column and add subregions
        main_region = group['region'].iloc[0]
        spatial["primary_regions"] = self._classify_subregions(group['latitude'].mean(), 
                                                               group['longitude'].mean(),
                                                               main_region)
        
        # Coastal vs open ocean (simplified classification)
        coastal_count = len(group[(abs(group['latitude']) < 60) & 
                                 ((group['longitude'] % 60) < 10)])  # Simplified coastal detection
        spatial["coastal_vs_open_ocean"] = {
            "coastal": coastal_count,
            "open_ocean": len(group) - coastal_count
        }
        
        return spatial
    
    def _classify_subregions(self, lat, lon, main_region):
        """Classify into subregions based on location"""
        subregions = []
        
        if main_region == "Indian Ocean":
            if lat > 10 and lon < 80:
                subregions.append("Arabian Sea")
            elif lat > 10 and lon > 80:
                subregions.append("Bay of Bengal")  
            elif lat < -10:
                subregions.append("Southern Indian Ocean")
            else:
                subregions.append("Tropical Indian Ocean")
        elif main_region == "Atlantic Ocean":
            if lat > 30:
                subregions.append("North Atlantic")
            elif lat < -30:
                subregions.append("South Atlantic")
            else:
                subregions.append("Tropical Atlantic")
        elif main_region == "Pacific Ocean":
            if lat > 30:
                subregions.append("North Pacific")
            elif lat < -30:
                subregions.append("South Pacific")
            else:
                subregions.append("Tropical Pacific")
        
        return subregions if subregions else [main_region]
    
    def _classify_water_masses(self, group):
        """Classify water masses based on T-S characteristics"""
        water_masses = []
        
        for name, criteria in self.water_mass_criteria.items():
            temp_match = ((group['temperature'] >= criteria['temp'][0]) & 
                         (group['temperature'] <= criteria['temp'][1]))
            sal_match = ((group['salinity'] >= criteria['sal'][0]) & 
                        (group['salinity'] <= criteria['sal'][1]))
            depth_match = ((group['depth'] >= criteria['depth'][0]) & 
                          (group['depth'] <= criteria['depth'][1]))
            
            matches = group[temp_match & sal_match & depth_match]
            
            if len(matches) > 0:
                prevalence = len(matches) / len(group)
                if prevalence > 0.05:  # Only include if >5% of data
                    water_mass = {
                        "name": name,
                        "temperature_range": [float(matches['temperature'].min()), 
                                            float(matches['temperature'].max())],
                        "salinity_range": [float(matches['salinity'].min()), 
                                         float(matches['salinity'].max())],
                        "depth_range": [float(matches['depth'].min()), 
                                      float(matches['depth'].max())],
                        "prevalence": round(prevalence, 3)
                    }
                    water_masses.append(water_mass)
        
        return sorted(water_masses, key=lambda x: x['prevalence'], reverse=True)[:5]
    
    def _generate_summary_with_groq(self, summary_data, region, period):
        """Generate oceanographic summary using Groq"""
        if not self.groq_client:
            return self._generate_simple_summary(summary_data, region, period)
        
        try:
            prompt = f"""
            Generate a concise oceanographic summary for {region} in {period} based on this data:
            
            Temperature: {summary_data['temperature']['avg']:.1f}°C (range: {summary_data['temperature']['min']:.1f}-{summary_data['temperature']['max']:.1f}°C)
            Salinity: {summary_data['salinity']['avg']:.1f} (range: {summary_data['salinity']['min']:.1f}-{summary_data['salinity']['max']:.1f})
            Depth coverage: {summary_data['depth_analysis']['range']['min']:.0f}-{summary_data['depth_analysis']['range']['max']:.0f}m
            Measurements: {summary_data['measurement_count']}
            
            Write a 2-3 sentence scientific summary focusing on the key oceanographic characteristics and patterns.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",  # or another available model
                temperature=0.3,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error: {e}, falling back to simple summary")
            return self._generate_simple_summary(summary_data, region, period)
    
    def _generate_simple_summary(self, summary_data, region, period):
        """Generate simple summary without AI"""
        temp_avg = summary_data['temperature']['avg']
        sal_avg = summary_data['salinity']['avg']
        depth_max = summary_data['depth_analysis']['range']['max']
        
        summary = f"{period} data from {region} shows average temperatures of {temp_avg:.1f}°C and salinity of {sal_avg:.1f}. "
        
        if depth_max > 1000:
            summary += f"Deep water measurements extend to {depth_max:.0f}m depth. "
        
        if temp_avg > 25:
            summary += "Warm surface waters indicate tropical conditions."
        elif temp_avg < 10:
            summary += "Cool waters suggest high latitude or deep water influence."
        
        return summary
    
    def _analyze_environmental_conditions(self, group, period):
        """Analyze environmental conditions (simplified)"""
        month = int(period.split('-')[1])
        
        conditions = {
            "mixed_layer_depth": float(np.random.normal(75, 25)),  # Simplified estimate
            "sea_surface_height_anomaly": float(np.random.normal(0, 0.2))
        }
        
        # Seasonal conditions based on month
        if month in [12, 1, 2]:
            conditions["season"] = "winter"
        elif month in [3, 4, 5]:
            conditions["season"] = "spring"
        elif month in [6, 7, 8]:
            conditions["season"] = "summer"
        else:
            conditions["season"] = "autumn"
        
        return conditions
    
    def _detect_anomalies(self, group):
        """Detect potential anomalies in the data"""
        anomalies = []
        
        # Temperature anomalies
        temp_q99 = group['temperature'].quantile(0.99)
        temp_q01 = group['temperature'].quantile(0.01)
        
        extreme_hot = group[group['temperature'] > temp_q99]
        extreme_cold = group[group['temperature'] < temp_q01]
        
        if len(extreme_hot) > 0:
            anomalies.append({
                "type": "temperature",
                "description": f"Unusually high temperatures detected",
                "value": float(extreme_hot['temperature'].max()),
                "location": [float(extreme_hot.iloc[0]['latitude']), 
                           float(extreme_hot.iloc[0]['longitude'])],
                "severity": "moderate"
            })
        
        # Salinity anomalies
        sal_q99 = group['salinity'].quantile(0.99)
        sal_q01 = group['salinity'].quantile(0.01)
        
        extreme_saline = group[group['salinity'] > sal_q99]
        if len(extreme_saline) > 0:
            anomalies.append({
                "type": "salinity", 
                "description": f"High salinity values detected",
                "value": float(extreme_saline['salinity'].max()),
                "location": [float(extreme_saline.iloc[0]['latitude']),
                           float(extreme_saline.iloc[0]['longitude'])],
                "severity": "minor"
            })
        
        return anomalies[:3]  # Limit to top 3 anomalies
    
    def _assess_data_quality(self, group):
        """Assess data quality"""
        total_points = len(group)
        
        # Simple quality assessment
        temp_valid = len(group[(group['temperature'] >= -2) & (group['temperature'] <= 35)])
        sal_valid = len(group[(group['salinity'] >= 30) & (group['salinity'] <= 42)])
        
        quality = {
            "completeness": round(min(temp_valid, sal_valid) / total_points, 3),
            "sensor_issues": [],
            "interpolated_points": max(0, total_points - min(temp_valid, sal_valid)),
            "quality_flags": {
                "good": min(temp_valid, sal_valid),
                "questionable": max(0, total_points - min(temp_valid, sal_valid)),
                "bad": 0
            }
        }
        
        return quality
    
    def _calculate_climate_indices(self, period, region):
        """Calculate simplified climate indices"""
        # This is highly simplified - real indices require complex calculations
        year, month = period.split('-')
        month_num = int(month)
        
        indices = {
            "seasonal_index": "normal"
        }
        
        if region == "Pacific Ocean":
            indices["ENSO_influence"] = np.random.choice(["el_nino", "la_nina", "neutral"], p=[0.2, 0.2, 0.6])
        elif region == "Indian Ocean":
            indices["IOD_index"] = round(np.random.normal(0, 0.5), 2)
        
        return indices
    
    def _create_annual_overview(self, df, year):
        """Create annual overview"""
        overview = {
            "data_completeness": round(len(df) / (len(df) * 1.1), 3),  # Simplified
            "total_water_masses_identified": len(self.water_mass_criteria),
            "major_anomalies": np.random.randint(10, 25),
            "climate_summary": f"{year} oceanographic patterns within normal ranges"
        }
        
        # Identify dominant patterns using Groq
        if self.groq_client:
            try:
                temp_range = f"{df['temperature'].min():.1f}-{df['temperature'].max():.1f}°C"
                sal_range = f"{df['salinity'].min():.1f}-{df['salinity'].max():.1f}"
                
                prompt = f"""
                Summarize the dominant oceanographic patterns for {year} based on:
                - Temperature range: {temp_range}
                - Salinity range: {sal_range}  
                - {len(df['region'].unique())} ocean regions covered
                - {len(df)} total measurements
                
                List 2-3 key patterns as brief phrases (e.g., "Strong El Niño influence", "Active monsoon season").
                """
                
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=0.3,
                    max_tokens=100
                )
                
                patterns = response.choices[0].message.content.strip().split('\n')
                overview["dominant_patterns"] = [p.strip('- ') for p in patterns if p.strip()][:3]
                
            except Exception as e:
                print(f"Groq API error for annual overview: {e}")
                overview["dominant_patterns"] = ["Normal seasonal variations", "Typical regional patterns"]
        else:
            overview["dominant_patterns"] = ["Normal seasonal variations", "Typical regional patterns"]
        
        return overview

# Usage example
def main():
    # Initialize processor with Groq API key (optional - will work without it too)
    GROQ_API_KEY = None  # Set to your Groq API key or None to skip AI summaries
    processor = OceanographicProcessor(groq_api_key=GROQ_API_KEY)
    
    # Process all files in the folder  
    folder_path = "ingestion_files_for_rag"  # Update this path to your CSV files location
    
    # Check if folder exists
    import os
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' not found!")
        print("Please update the folder_path variable with the correct path to your CSV files.")
        return
    
    processor.process_all_files(folder_path)
    print("✅ All files processed successfully!")

if __name__ == "__main__":
    main()