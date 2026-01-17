"""
Automated Trading Scheduler
Runs the AI pipeline at regular intervals (e.g., every hour)
"""

import schedule
import time
import logging
from datetime import datetime
import subprocess
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ================================================================
# CONFIGURATION
# ================================================================

class SchedulerConfig:
    # Schedule settings
    RUN_INTERVAL_MINUTES = 60  # Run every 60 minutes
    
    # Script to run
    MAIN_SCRIPT = "main.py"
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 300  # 5 minutes
    
    # Health check
    ENABLE_HEALTH_CHECK = True
    HEALTH_CHECK_INTERVAL = 15  # Check every 15 minutes

# ================================================================
# SCHEDULER CLASS
# ================================================================

class TradingScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.last_run_status = None
        self.consecutive_failures = 0
        self.total_runs = 0
        self.successful_runs = 0
        
    def run_pipeline(self):
        """Execute the main trading pipeline"""
        self.total_runs += 1
        
        logger.info("="*60)
        logger.info(f"🚀 STARTING AI PIPELINE - Run #{self.total_runs}")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        try:
            # Run the main script
            result = subprocess.run(
                [sys.executable, self.config.MAIN_SCRIPT],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
                logger.info("Output:")
                logger.info(result.stdout)
                
                self.last_run_status = "success"
                self.consecutive_failures = 0
                self.successful_runs += 1
                
            else:
                logger.error("❌ PIPELINE FAILED")
                logger.error(f"Return code: {result.returncode}")
                logger.error("Error output:")
                logger.error(result.stderr)
                
                self.last_run_status = "failed"
                self.consecutive_failures += 1
                
                # Alert if too many failures
                if self.consecutive_failures >= 3:
                    self.send_alert(f"⚠️ ALERT: {self.consecutive_failures} consecutive failures!")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ PIPELINE TIMEOUT (30 minutes)")
            self.last_run_status = "timeout"
            self.consecutive_failures += 1
            
        except Exception as e:
            logger.error(f"❌ UNEXPECTED ERROR: {str(e)}")
            self.last_run_status = "error"
            self.consecutive_failures += 1
        
        finally:
            self.print_statistics()
    
    def health_check(self):
        """Perform system health check"""
        logger.info("🏥 Health Check...")
        
        # Check if main script exists
        import os
        if not os.path.exists(self.config.MAIN_SCRIPT):
            logger.error(f"❌ Main script not found: {self.config.MAIN_SCRIPT}")
            return False
        
        # Check Python dependencies
        try:
            import pandas
            import numpy
            import sklearn
            from supabase import create_client
            logger.info("✅ All dependencies installed")
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            return False
        
        # Check consecutive failures
        if self.consecutive_failures >= 5:
            logger.warning(f"⚠️ WARNING: {self.consecutive_failures} consecutive failures")
            return False
        
        logger.info(f"✅ Health check passed (Success rate: {self.get_success_rate():.1f}%)")
        return True
    
    def get_success_rate(self):
        """Calculate success rate"""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100
    
    def print_statistics(self):
        """Print scheduler statistics"""
        logger.info("─" * 60)
        logger.info("📊 SCHEDULER STATISTICS")
        logger.info(f"   Total runs:          {self.total_runs}")
        logger.info(f"   Successful runs:     {self.successful_runs}")
        logger.info(f"   Failed runs:         {self.total_runs - self.successful_runs}")
        logger.info(f"   Success rate:        {self.get_success_rate():.1f}%")
        logger.info(f"   Consecutive failures: {self.consecutive_failures}")
        logger.info(f"   Last status:         {self.last_run_status}")
        logger.info("─" * 60)
    
    def send_alert(self, message: str):
        """Send alert (implement your preferred method)"""
        logger.critical(message)
        # TODO: Implement email/SMS/Telegram alerts
        # Example:
        # send_email(to="your@email.com", subject="Trading Alert", body=message)
        # send_telegram(chat_id="your_chat_id", text=message)

# ================================================================
# MAIN SCHEDULER SETUP
# ================================================================

def main():
    logger.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       AI TRADING SYSTEM - AUTOMATED SCHEDULER             ║
    ║             Running pipeline every hour                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    config = SchedulerConfig()
    scheduler = TradingScheduler(config)
    
    # Schedule the pipeline to run at regular intervals
    schedule.every(config.RUN_INTERVAL_MINUTES).minutes.do(scheduler.run_pipeline)
    
    # Schedule health checks
    if config.ENABLE_HEALTH_CHECK:
        schedule.every(config.HEALTH_CHECK_INTERVAL).minutes.do(scheduler.health_check)
    
    # Run immediately on startup
    logger.info("🚀 Running initial pipeline execution...")
    scheduler.run_pipeline()
    
    # Keep running
    logger.info(f"⏰ Scheduler started - Running every {config.RUN_INTERVAL_MINUTES} minutes")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        logger.info("\n⛔ Scheduler stopped by user")
        scheduler.print_statistics()
        logger.info("Goodbye!")

# ================================================================
# ALTERNATIVE: Manual Schedule Configuration
# ================================================================

def setup_advanced_schedule(scheduler: TradingScheduler):
    """
    Advanced scheduling examples
    Uncomment and modify as needed
    """
    
    # Run every hour
    schedule.every().hour.do(scheduler.run_pipeline)
    
    # Run at specific times
    # schedule.every().day.at("09:00").do(scheduler.run_pipeline)  # 9 AM
    # schedule.every().day.at("15:00").do(scheduler.run_pipeline)  # 3 PM
    # schedule.every().day.at("21:00").do(scheduler.run_pipeline)  # 9 PM
    
    # Run every N hours during trading hours
    # schedule.every(4).hours.do(scheduler.run_pipeline)
    
    # Run on specific days
    # schedule.every().monday.at("09:00").do(scheduler.run_pipeline)
    # schedule.every().friday.at("17:00").do(scheduler.run_pipeline)
    
    # Health check every 15 minutes
    schedule.every(15).minutes.do(scheduler.health_check)

# ================================================================
# DEPLOYMENT HELPERS
# ================================================================

def create_systemd_service():
    """
    Generate systemd service file for Linux
    Save as /etc/systemd/system/trading-ai.service
    """
    service = """[Unit]
Description=AI Trading System Scheduler
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/trading-ai
ExecStart=/usr/bin/python3 /path/to/trading-ai/scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    print("Linux Systemd Service:")
    print(service)
    print("\nSetup commands:")
    print("sudo cp trading-ai.service /etc/systemd/system/")
    print("sudo systemctl enable trading-ai")
    print("sudo systemctl start trading-ai")
    print("sudo systemctl status trading-ai")

def create_windows_task():
    """
    Instructions for Windows Task Scheduler
    """
    instructions = """
Windows Task Scheduler Setup:
==============================

1. Open Task Scheduler (taskschd.msc)

2. Create Basic Task:
   - Name: AI Trading Scheduler
   - Description: Runs AI trading pipeline every hour

3. Trigger:
   - Daily
   - Start: Tomorrow at 00:00
   - Recur every: 1 day
   - Repeat task every: 1 hour
   - For a duration of: Indefinitely

4. Action:
   - Start a program
   - Program: C:\\Python39\\python.exe
   - Arguments: C:\\path\\to\\scheduler.py
   - Start in: C:\\path\\to\\trading-ai

5. Conditions:
   - Uncheck "Start only if on AC power"
   - Check "Wake computer to run task"

6. Settings:
   - Allow task to run on demand
   - Run task as soon as possible after missed start
   - If task fails, restart every: 10 minutes
    """
    
    print(instructions)

def create_cron_job():
    """
    Generate crontab entry for Linux/Mac
    """
    cron = """
# Add to crontab with: crontab -e

# Run every hour at minute 0
0 * * * * cd /path/to/trading-ai && /usr/bin/python3 scheduler.py >> logs/cron.log 2>&1

# Or run every 30 minutes
*/30 * * * * cd /path/to/trading-ai && /usr/bin/python3 scheduler.py >> logs/cron.log 2>&1
    """
    
    print("Cron Job Configuration:")
    print(cron)

# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
    
    # Uncomment to see deployment helpers
    # create_systemd_service()
    # create_windows_task()
    # create_cron_job()
