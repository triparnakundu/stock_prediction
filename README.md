# stock_prediction
This project focuses on the replacement of individual stocks using ETFs and how it helps predicting stock movements.

**Predicting Daily Stock Movements with Deep Learning Models**
In this paper, deep learning models like LSTM,CNN and RNN were explored to predict the stock movement prices for Exchange Traded Funds (ETFs)and S&P 500 for over a ”24-hour” close-to-close trading session. To simplify the input of the model and improve its efficiency, individual stock holdings were replaced with their corresponding ETFs. To justify the reasons for this replacement, results that demonstrated a very strong correlation averaging 0.99 and a low hamming distance of 0.25 across both input and output sequences were achieved, confirming their validity. To evaluate the efficiency of the model, historical data were considered spanning a 25-year period (2000 to 2024), and it was observed that LSTM consistently outperformed CNN and RNN, with an average improvement of 48.59% and 90.50%, respectively.
Moreover, LSTM also performed better than the Buy and Hold strategy by 53. 59%, highlighting its ability to make more profitable decisions than a passive investment strategy. These results suggest that LSTM has the potential to improve stock prediction and investment strategies and can replace traditional methods or other deep learning methods for financial forecasting.
The paper also highlights the important questions:
**1. Can ETFs Replace Individual Stock Movement as a Centroid?
2. Expected Performance of Stock Prediction Using Deep Learning Methods**

**Step-by-Step Execution of the Project:**
Here's a detailed breakdown of the steps to execute the ETF prediction and analysis project efficiently:

This project requires a significant amount of storage space, so ensure you have sufficient disk space available before proceeding or make use of compute.

**Step 1: Download ETF Data**
- This script downloads historical ETF data spanning the last 25 years. The timeframe can be modified based on requirements.  
- It fetches data for various ETFs and saves them as CSV files, which will later be used for analysis and predictions.  

**Execution Steps:**  
1. Open a terminal or command prompt in the directory containing `ETF_csv.py`.  
2. Run the script using the command:  
   ```bash
   python ETF_csv.py
   ```
3. Ensure all ETF data files are successfully downloaded before proceeding to the next step.  

**Step 2: Predict Returns Using Machine Learning Models**
- This code utilizes the downloaded ETF data to predict future returns using different lookback values and machine learning models, including LSTM, CNN, and RNN.  
- It calculates the predicted returns and accuracy for each model and saves the results in a separate CSV file.  
- A new CSV file containing the predicted returns and accuracy metrics for various models.  

**Execution Steps:**  
1. Ensure that all required dependencies (TensorFlow, NumPy, Pandas, etc.) are installed.  
2. Run the script using the command:  
   ```bash
   python ETF_predicted_returns.py
   ```
3. The script will process the downloaded ETF CSV files, compute predictions, and store the results.  

**Step 3: Model Evaluation and Performance Analysis**
- This Jupyter Notebook analyzes the predictive performance of different models over the years using close-to-close predicted returns.  
- It includes correlation and Hamming distance calculations for all ETFs and SPY for both input and output sequences.  
- This validates our reason for replacing ETFs with actual stock holdings.
**Execution Steps:**  
1. Open the Jupyter Notebook using the command:  
   ```bash
   jupyter notebook
   ```
2. Load the `new_analysis.ipynb` file and execute each cell sequentially.  
3. The notebook will generate visualizations and statistical comparisons of the models.  

## **Step 4: Volatility Analysis** 
- This notebook examines ETF volatility over time by comparing model-generated predictions with actual market volatility.  
- It calculates key financial metrics, including:  
  - **Annual Returns**  
  - **Sharpe Ratio** (risk-adjusted returns)  
  - **Maximum Drawdown (MDD)**  
  - **Number of switches over time for every model**  

**Execution Steps:**  
1. Open the Jupyter Notebook as before and load `volatility.ipynb`.  
2. Execute each cell step by step to compute volatility comparisons.  

**Step 5: Final Performance Comparison** 
- This notebook consolidates findings from previous steps, providing a comprehensive performance comparison of all ETFs over the years.  
- It presents graphical comparisons and numerical summaries for easier interpretation of results.  
- Final performance comparison values, trends, and visualization charts for all ETFs over multiple years are displayed.
**Execution Steps:**  
1. Open and execute `analysis.ipynb` in Jupyter Notebook.  
2. Review the final comparisons and insights drawn from the different models.  

Ensure all scripts and notebooks execute correctly before interpreting the results. 🚀
