#!/usr/bin/env python3
"""
German Solar Generation Data Collection Pipeline v3
===================================================

This script orchestrates the complete pipeline for collecting and processing
quarter-hourly German solar generation data from the ENTSO-E Transparency Platform
using the new RESTful API.

Based on official documentation:
https://transparencyplatform.zendesk.com/hc/en-us/articles/15692855254548-Sitemap-for-Restful-API-Integration

IMPORTANT: This version uses the new RESTful API structure and includes fallback 
data generation when the ENTSO-E API is unavailable.

The pipeline includes:
1. Data collection from ENTSO-E RESTful API (with fallback)
2. Data validation and quality checks
3. Data processing and feature engineering
4. Storage in tidy Parquet format for modeling

Usage:
    python collect_german_solar_generation_v3.py --start-year 2018 --end-year 2020
    python collect_german_solar_generation_v3.py --test  # Run with test data
"""

import sys
import argparse
import time
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from entsoe_data_collector_v3 import ENTSOESolarCollectorV3
from solar_generation_processor import SolarGenerationProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GermanSolarGenerationPipelineV3:
    """
    German Solar Generation Data Collection Pipeline v3
    
    This version uses the new RESTful API structure based on Zendesk documentation.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        output_dir : Path, optional
            Output directory for data storage
        """
        self.output_dir = output_dir or Path("data_german_solar_generation")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.collector = ENTSOESolarCollectorV3(output_dir=self.output_dir)
        self.processor = SolarGenerationProcessor(
            input_dir=self.output_dir / "processed",
            output_dir=self.output_dir / "modeling"
        )
        
        logger.info(f"German Solar Generation Pipeline v3 initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Using new RESTful API based on Zendesk documentation")
    
    def collect_data_for_period(self, start_year: int, end_year: int, test_mode: bool = False) -> pd.DataFrame:
        """
        Collect solar generation data for the specified period.
        
        Parameters:
        -----------
        start_year : int
            Start year for data collection
        end_year : int
            End year for data collection
        test_mode : bool
            If True, collect only a small sample for testing
            
        Returns:
        --------
        pd.DataFrame
            Collected solar generation data
        """
        logger.info(f"Starting data collection for {start_year}-{end_year}")
        
        if test_mode:
            logger.info("Running in test mode - collecting small sample")
            # Collect just one week for testing
            start_date = datetime(start_year, 1, 1)
            end_date = start_date + timedelta(days=7)
            
            df = self.collector.fetch_solar_generation(start_date, end_date)
            logger.info(f"Test data collection completed: {len(df)} records")
            return df
        else:
            # Collect data month by month
            dataframes = self.collector.fetch_data_by_month(start_year, end_year)
            
            if dataframes:
                # Combine all dataframes
                combined_df = pd.concat(dataframes, ignore_index=True)
                logger.info(f"Data collection completed: {len(combined_df)} total records")
                return combined_df
            else:
                logger.warning("No data collected")
                return pd.DataFrame()
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate the collected data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data to process
            
        Returns:
        --------
        pd.DataFrame
            Processed data
        """
        if len(df) == 0:
            logger.warning("No data to process")
            return df
        
        logger.info("Processing collected data...")
        
        # Basic validation
        logger.info(f"Input data: {len(df)} records")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Data source: {df['data_source'].iloc[0] if len(df) > 0 else 'unknown'}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'])
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Processing completed: {len(df)} records")
        return df
    
    def save_data(self, df: pd.DataFrame, start_year: int, end_year: int, test_mode: bool = False) -> List[Path]:
        """
        Save processed data to various formats.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to save
        start_year : int
            Start year for filename
        end_year : int
            End year for filename
        test_mode : bool
            If True, use test prefix for filenames
            
        Returns:
        --------
        List[Path]
            List of saved file paths
        """
        if len(df) == 0:
            logger.warning("No data to save")
            return []
        
        saved_files = []
        
        # Generate filename
        prefix = "test_" if test_mode else ""
        base_filename = f"{prefix}german_solar_generation_v3_{start_year}_{end_year}"
        
        # Save as Parquet
        parquet_file = self.collector.save_to_parquet(df, f"{base_filename}.parquet")
        saved_files.append(parquet_file)
        
        # Save as CSV for compatibility
        csv_file = self.output_dir / "processed" / f"{base_filename}.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)
        logger.info(f"Data saved to {csv_file}")
        
        # Save summary statistics
        summary = self.collector.get_data_summary(df)
        summary_file = self.output_dir / "processed" / f"{base_filename}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files.append(summary_file)
        logger.info(f"Summary saved to {summary_file}")
        
        return saved_files
    
    def run_complete_pipeline(self, start_year: int, end_year: int, test_mode: bool = False) -> Dict:
        """
        Run the complete data collection and processing pipeline.
        
        Parameters:
        -----------
        start_year : int
            Start year for data collection
        end_year : int
            End year for data collection
        test_mode : bool
            If True, run in test mode with limited data
            
        Returns:
        --------
        Dict
            Pipeline results and summary
        """
        logger.info("="*80)
        logger.info("🚀 Starting German Solar Generation Pipeline v3")
        logger.info("="*80)
        logger.info(f"Period: {start_year}-{end_year}")
        logger.info(f"Mode: {'Test' if test_mode else 'Production'}")
        logger.info("Using new RESTful API based on Zendesk documentation")
        logger.info("="*80)
        
        try:
            # Step 1: Collect data
            logger.info("📊 Step 1: Collecting data from ENTSO-E RESTful API...")
            df = self.collect_data_for_period(start_year, end_year, test_mode)
            
            if len(df) == 0:
                logger.error("No data collected. Pipeline failed.")
                return {"success": False, "error": "No data collected"}
            
            # Step 2: Process data
            logger.info("🔧 Step 2: Processing and validating data...")
            processed_df = self.process_data(df)
            
            # Step 3: Save data
            logger.info("💾 Step 3: Saving data to files...")
            saved_files = self.save_data(processed_df, start_year, end_year, test_mode)
            
            # Step 4: Generate summary
            logger.info("📈 Step 4: Generating summary statistics...")
            summary = self.collector.get_data_summary(processed_df)
            
            # Final results
            results = {
                "success": True,
                "start_year": start_year,
                "end_year": end_year,
                "test_mode": test_mode,
                "total_records": len(processed_df),
                "data_source": summary.get("data_source", "unknown"),
                "date_range": summary.get("date_range", {}),
                "generation_stats": summary.get("generation_stats", {}),
                "saved_files": [str(f) for f in saved_files],
                "output_directory": str(self.output_dir)
            }
            
            logger.info("="*80)
            logger.info("✅ Pipeline completed successfully!")
            logger.info("="*80)
            logger.info(f"📊 Total records: {results['total_records']}")
            logger.info(f"📅 Date range: {results['date_range'].get('start', 'N/A')} to {results['date_range'].get('end', 'N/A')}")
            logger.info(f"🔌 Data source: {results['data_source']}")
            logger.info(f"💾 Files saved: {len(results['saved_files'])}")
            logger.info(f"📁 Output directory: {results['output_directory']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_data_report(self, results: Dict) -> str:
        """
        Generate a human-readable report of the pipeline results.
        
        Parameters:
        -----------
        results : Dict
            Pipeline results
            
        Returns:
        --------
        str
            Formatted report
        """
        if not results.get("success", False):
            return f"❌ Pipeline failed: {results.get('error', 'Unknown error')}"
        
        report = f"""
{'='*80}
🇩🇪 German Solar Generation Data Collection Report v3
{'='*80}

📊 Collection Summary:
   • Period: {results['start_year']}-{results['end_year']}
   • Mode: {'Test' if results['test_mode'] else 'Production'}
   • Total Records: {results['total_records']:,}
   • Data Source: {results['data_source']}

📅 Date Range:
   • Start: {results['date_range'].get('start', 'N/A')}
   • End: {results['date_range'].get('end', 'N/A')}

⚡ Generation Statistics:
   • Mean: {results['generation_stats'].get('mean_mw', 0):.2f} MW
   • Maximum: {results['generation_stats'].get('max_mw', 0):.2f} MW
   • Minimum: {results['generation_stats'].get('min_mw', 0):.2f} MW
   • Standard Deviation: {results['generation_stats'].get('std_mw', 0):.2f} MW

💾 Output Files ({len(results['saved_files'])} files):
"""
        
        for i, file_path in enumerate(results['saved_files'], 1):
            report += f"   {i}. {file_path}\n"
        
        report += f"""
📁 Output Directory: {results['output_directory']}

🔧 Technical Details:
   • API: New RESTful API based on Zendesk documentation
   • Format: Quarter-hourly (15-minute) resolution
   • Storage: Parquet and CSV formats
   • Processing: Validated and cleaned data

✅ Status: Pipeline completed successfully!
{'='*80}
"""
        
        return report


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="German Solar Generation Data Collection Pipeline v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (small sample)
  python collect_german_solar_generation_v3.py --test
  
  # Full collection 2018-2020
  python collect_german_solar_generation_v3.py --start-year 2018 --end-year 2020
  
  # Custom period
  python collect_german_solar_generation_v3.py --start-year 2019 --end-year 2021
  
  # Custom output directory
  python collect_german_solar_generation_v3.py --start-year 2018 --end-year 2020 --output-dir ./my_data
        """
    )
    
    parser.add_argument(
        "--start-year", 
        type=int, 
        default=2018,
        help="Start year for data collection (default: 2018)"
    )
    
    parser.add_argument(
        "--end-year", 
        type=int, 
        default=2020,
        help="End year for data collection (default: 2020)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run in test mode with limited data"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Output directory for data storage"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_year > args.end_year:
        print("❌ Error: Start year must be less than or equal to end year")
        sys.exit(1)
    
    if args.start_year < 2015:
        print("❌ Error: Start year must be 2015 or later (ENTSO-E data availability)")
        sys.exit(1)
    
    # Initialize pipeline
    output_dir = Path(args.output_dir) if args.output_dir else None
    pipeline = GermanSolarGenerationPipelineV3(output_dir=output_dir)
    
    try:
        # Run pipeline
        summary = pipeline.run_complete_pipeline(
            start_year=args.start_year,
            end_year=args.end_year,
            test_mode=args.test
        )
        
        # Generate and print report
        report = pipeline.generate_data_report(summary)
        print(report)
        
        # Save report to file
        report_file = pipeline.output_dir / "data_collection_report_v3.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_file}")
        
        # Exit with appropriate code
        if summary.get("success", False):
            print("\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Pipeline failed: {summary.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
