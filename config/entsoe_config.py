# ENTSO-E API Configuration
# ========================
#
# This file contains configuration for the ENTSO-E Transparency Platform API.
# 
# To use this file:
# 1. Register for API access at https://transparency.entsoe.eu/
# 2. Get your API key from the platform
# 3. Replace the placeholder below with your actual API key
# 4. Keep this file secure and don't commit it to version control

# Your ENTSO-E API key
# Replace this with your actual API key from https://transparency.entsoe.eu/
API_KEY = "your_entsoe_api_key_here"

# Alternative: You can also set the API key as an environment variable:
# export ENTSOE_API_KEY="your_actual_api_key"

# API Configuration
# The new ENTSO-E API uses a different base URL structure
API_BASE_URL = "https://transparency.entsoe.eu/api/v1"
REQUEST_TIMEOUT = 300  # seconds
RATE_LIMIT_DELAY = 1   # seconds between requests

# Germany bidding zone code
GERMANY_BIDDING_ZONE = "10Y1001A1001A83F"

# Solar generation PSR type
SOLAR_PSR_TYPE = "B16"

# Data collection parameters
DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2020
