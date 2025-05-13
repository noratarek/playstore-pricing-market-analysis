# Day 1: Initial Data Assessment Findings

## Dataset Overview
- Dataset dimensions: 10,841 rows x 13 columns
- Time period: Data from 2018 (based on Last Updated dates)
- Source: Google Play Store app data

## Key Findings

### Data Quality Issues
- Missing values: 
  - Rating: 13.60% missing
  - Current Ver: 0.07% missing
  - Android Ver: 0.03% missing
  - Content Rating and Type: 0.01% missing
- Data type inconsistencies:
  - Reviews, Installs, Price and Size are stored as strings with inconsistent formats
  - Size uses mixed units (M, k)
  - Installs contains commas and '+' characters
  - Price contains '$' prefix
- Potential outliers:
  - Rating has some suspicious values (max 19.0 when the scale should be 1-5)

### Pricing Analysis Initial Insights
- Distribution of free vs. paid apps:
  - Free apps: 92.6% (10,040 apps)
  - Paid apps: 7.4% (800 apps)
- Price ranges:
  - Most common paid app prices: $0.99, $2.99, $1.99, $4.99
  - Price categories defined: Free, Low-cost ($0.01-$1), Mid-price ($1-$5), High-price ($5-$10), Premium ($10+)
- Highest priced categories:
  - Medical, Business, and Education apps tend to have the highest average prices
  - Entertainment and Social apps tend to be more frequently free

### Market Saturation Initial Insights
- Most crowded app categories:
  - Family (1,972 apps)
  - Game (1,144 apps)
  - Tools (843 apps)
- Categories with highest total installs:
  - Games, Communication, and Social tend to have the highest total installs
- Saturation Index findings:
  - Categories with high competition relative to install base (saturated): Medical, Business, Education
  - Potential opportunity gaps: Categories with high install rates but relatively fewer apps

## Next Steps
1. **Data Cleaning**:
   - Convert all string columns to appropriate numeric types
   - Handle missing values in the Rating column
   - Remove or cap outliers in Rating and other numeric columns
   - Standardize formats for Size, Installs, and Price

2. **Feature Engineering**:
   - Create price categories for analysis
   - Calculate metrics for market saturation
   - Derive app age from Last Updated date (if needed)

3. **Further Analysis Questions**:
   - What is the relationship between price and app rating?
   - Which categories show better monetization potential?
   - Are there under-served app categories with high demand?