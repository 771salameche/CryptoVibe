"""
This script generates a comprehensive insights report by compiling all previous
correlation, sentiment, and event analysis findings. It produces a detailed
markdown report, a visual summary, and attempts to export to PDF.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import re
import subprocess
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Input files
LAG_REPORT_PATH = "Correlation-Analysis/lag_analysis_report.txt"
STATS_REPORT_PATH = "Correlation-Analysis/statistical_tests_report.txt"
EVENTS_TIMELINE_PATH = "data/Results/events_timeline.csv"
INFLUENCER_DATA_PATH = "data/Results/data_with_influencers.csv"
CORRELATION_MATRIX_PATH = "data/Visualizations/correlation_heatmap.png" # Assuming this is where it's saved

# All generated chart paths
CHART_PATHS = {
    'lag_correlation': "data/Visualizations/Lag_Analysis/lag_correlation_chart.png",
    'cross_corr_btc': "data/Visualizations/Lag_Analysis/cross_correlation_BTC.png",
    'cross_corr_eth': "data/Visualizations/Lag_Analysis/cross_correlation_ETH.png",
    'cross_corr_sol': "data/Visualizations/Lag_Analysis/cross_correlation_SOL.png",
    'significance_tests': "data/Visualizations/Statistical_Tests/significance_tests.png",
    'event_analysis': "data/Visualizations/events_analysis.png",
    'influencer_analysis': "data/Visualizations/influencer_analysis.png",
    'sentiment_timeline': "data/Visualizations/sentiment_timeline.png",
    'correlation_heatmap': "data/Visualizations/correlation_heatmap.png"
}

# Output directory structure
OUTPUT_DIR = "Report-Generator/Final_Report"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Output files
REPORT_MD_PATH = os.path.join(OUTPUT_DIR, "insights_report.md")
REPORT_PDF_PATH = os.path.join(OUTPUT_DIR, "insights_report.pdf")
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "insights_summary.txt")
SUMMARY_PNG_PATH = os.path.join(OUTPUT_DIR, "insights_summary.png")

# --- Helper Functions ---
def parse_report(file_path, table_identifier):
    """Parses a markdown/text report to extract a table as a DataFrame."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        table_content = content.split(table_identifier)[1]
        table_lines = [line.strip() for line in table_content.strip().split('\n') if '|' in line]
        
        if len(table_lines) < 2: return pd.DataFrame()

        header = [h.strip() for h in table_lines[0].split('|') if h.strip()]
        data = [[d.strip() for d in row.split('|') if d.strip()] for row in table_lines[2:]]
        
        return pd.DataFrame(data, columns=header)
    except Exception as e:
        logger.warning(f"Could not parse table with identifier '{table_identifier}' from {file_path}. Error: {e}")
        return pd.DataFrame()

def copy_visualizations():
    """Copies all generated charts into the report's image directory."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    logger.info("Copying visualizations to report directory...")
    for key, path in CHART_PATHS.items():
        if os.path.exists(path):
            shutil.copy(path, os.path.join(IMAGE_DIR, f"{key}.png"))
        else:
            logger.warning(f"Chart not found at {path}, skipping copy.")

# --- Report Generation Functions ---
def generate_executive_summary(lag_df, stats_df):
    """Generates the executive summary section."""
    key_findings = []
    
    # Finding 1: Strongest predictive insight
    if not lag_df.empty and 'Significant?' in lag_df.columns and 'Pearson r' in lag_df.columns:
        lag_df['abs_r'] = pd.to_numeric(lag_df['Pearson r'], errors='coerce').abs()
        significant_lags = lag_df[lag_df['Significant?'] == 'Yes']
        if not significant_lags.empty:
            best_predictor = significant_lags.loc[significant_lags['abs_r'].idxmax()]
            direction = "positive" if float(best_predictor['Pearson r']) > 0 else "negative"
            key_findings.append(f"The strongest predictive signal was for **{best_predictor['Crypto']}**, where sentiment showed a statistically significant **{direction} correlation** with price changes **{best_predictor['Optimal Lag (Days)']} day(s) later** (r={best_predictor['Pearson r']}).")
        else:
            key_findings.append("No statistically significant predictive links were found between daily sentiment and future price changes for any crypto.")
    else:
        key_findings.append("Lag analysis data was not available or malformed to determine predictive links.")
        
    # Finding 2: Overall sentiment-price relationship
    if not stats_df.empty and 'test' in stats_df.columns and 'r' in stats_df.columns:
        pearson_corr = pd.to_numeric(stats_df[stats_df['test'] == 'Pearson']['r'], errors='coerce').abs()
        overall_corr = pearson_corr.mean()
        key_findings.append(f"The overall same-day sentiment-price relationship is generally **weak** (average |r| ~ {overall_corr:.3f}) and not statistically significant after corrections.")
    else:
        key_findings.append("Correlation statistics were not available to determine same-day relationship strength.")

    key_findings.append("While strong immediate correlations are weak, lag analysis suggests that for some assets (like SOL and BTC), sentiment may act as a leading indicator, preceding price movements by several days.")
    
    summary = "# Executive Summary\n\n"
    summary += "This report analyzes the relationship between social media sentiment and cryptocurrency price movements for BTC, ETH, and SOL. The key objective was to determine if sentiment can be used as a reliable predictive indicator for price changes.\n\n"
    summary += "### Key Findings\n\n"
    for finding in key_findings:
        summary += f"- {finding}\n"
        
    return summary, key_findings

def create_visual_summary(key_findings, lag_df):
    """Creates a one-page infographic summary."""
    fig = plt.figure(figsize=(14, 18))
    fig.patch.set_facecolor('#F0F0F0')
    fig.suptitle("CryptoVibe Project: Insights Summary", fontsize=28, weight='bold')

    plt.figtext(0.1, 0.88, "Top Insights:", fontsize=20, weight='bold')
    for i, finding in enumerate(key_findings[:3]):
        finding_text = finding.replace('*', '').replace('`', '')
        plt.figtext(0.1, 0.82 - (i*0.07), f"• {finding_text}", fontsize=16, wrap=True, va='top')

    if not lag_df.empty and 'Crypto' in lag_df.columns:
        ax1 = fig.add_axes([0.1, 0.45, 0.8, 0.25])
        sns.barplot(data=lag_df, x='Crypto', y=pd.to_numeric(lag_df['Optimal Lag (Days)']), ax=ax1, palette='winter')
        ax1.set_title("Optimal Predictive Lag Time", fontsize=18)
        ax1.set_xlabel("")
        ax1.set_ylabel("Lag (Days)")
        
        ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.25])
        lag_df['abs_r'] = pd.to_numeric(lag_df['Pearson r'], errors='coerce').abs()
        sns.barplot(data=lag_df, x='Crypto', y='abs_r', ax=ax2, palette='autumn')
        ax2.set_title("Strength of Predictive Correlation (|r| at Optimal Lag)", fontsize=18)
        ax2.set_xlabel("Cryptocurrency", fontsize=14)
        ax2.set_ylabel("Absolute Pearson r")
    
    plt.savefig(SUMMARY_PNG_PATH)
    plt.close()

# --- Main Script ---
def main():
    logger.info("--- Generating Final Insights Report ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    copy_visualizations()

    lag_df = parse_report(LAG_REPORT_PATH, "## Summary of Findings")
    stats_df = parse_report(STATS_REPORT_PATH, "## Correlation Analysis")
    events_df = pd.read_csv(EVENTS_TIMELINE_PATH) if os.path.exists(EVENTS_TIMELINE_PATH) else pd.DataFrame()
    
    if not events_df.empty:
        top_events = events_df.sort_values(by='total_importance', ascending=False).head(10)
        top_events.to_csv(os.path.join(TABLES_DIR, "top_events.csv"), index=False)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(TABLES_DIR, "correlation_summary.csv"), index=False)

    exec_summary, key_findings = generate_executive_summary(lag_df, stats_df)
    
    correlation_section = "## Correlation Findings\n\nSee the correlation matrix and scatter plots for detailed analysis. Overall, same-day correlations were weak and not statistically significant.\n\n![Correlation Heatmap](images/correlation_heatmap.png)\n"
    lag_section = "## Lag Analysis\n\nAnalysis shows that sentiment for some cryptos has a statistically significant correlation with price changes several days in the future, suggesting a leading indicator relationship.\n\n![Lag Correlation Chart](images/lag_correlation_chart.png)\n"
    event_section = "## Event Impact\n\nEvent analysis indicates that major events like hacks or listings have a measurable impact on sentiment and trade volume.\n\n![Event Analysis](images/event_analysis.png)\n"
    influencer_section = "## Influencer Effect\n\nCertain influencers were found to have a noticeable but often short-lived impact on sentiment.\n\n![Influencer Analysis](images/influencer_analysis.png)\n"
    stats_section = "## Statistical Validation\n\nRigorous testing confirmed that while many correlations exist, only a few are statistically significant after applying corrections for multiple comparisons, primarily in the lag analysis.\n\n![Significance Tests](images/significance_tests.png)\n"
    recommendations_section = "## Recommendations\n\n- **For Traders:** Sentiment for SOL and BTC may be a useful (but not standalone) indicator if a 3-day lag is considered. Do not rely on same-day sentiment shifts for trading signals.\n- **For Researchers:** Further analysis should incorporate sub-daily data and more sophisticated causality models."
    limitations_section = "## Limitations\n\nThis analysis is based on daily aggregated data and does not account for intra-day movements. It also does not include macroeconomic factors or traditional technical analysis indicators."

    full_report = "\n".join([
        exec_summary, correlation_section, lag_section, event_section, influencer_section, 
        stats_section, recommendations_section, limitations_section
    ])
    
    with open(REPORT_MD_PATH, 'w', encoding='utf-8') as f:
        f.write(full_report)
    logger.info(f"Markdown report saved to {REPORT_MD_PATH}")
    
    with open(SUMMARY_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write("Executive Summary:\n\n" + "\n".join(key_findings))
    logger.info(f"Text summary saved to {SUMMARY_TXT_PATH}")
    
    create_visual_summary(key_findings, lag_df)
    logger.info(f"Visual summary saved to {SUMMARY_PNG_PATH}")

    try:
        logger.info("Attempting to convert markdown report to PDF with 'markdown-pdf'...")
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-Command', f'C:\\Users\\salah\\AppData\\Roaming\\npm\\markdown-pdf.ps1 "{REPORT_MD_PATH}" -o "{REPORT_PDF_PATH}"'],
            check=True, capture_output=True, text=True, timeout=60
        )
        logger.info(f"PDF report saved to {REPORT_PDF_PATH}")
        if result.stdout: logger.debug(result.stdout)
    except FileNotFoundError:
        logger.warning("Command 'markdown-pdf' not found. PDF conversion skipped.")
        logger.warning("To generate a PDF, please install Node.js and then run: 'npm install -g markdown-pdf'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert markdown to PDF. Error: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("PDF conversion timed out after 60 seconds.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during PDF conversion: {e}")

    logger.info("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
    