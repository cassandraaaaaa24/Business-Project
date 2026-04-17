import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SalesPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Prediction App")
        self.df = None
        self.model = None
        
        # GUI elements
        self.load_button = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.load_button.pack(pady=10)
        
        self.columns_label = tk.Label(root, text="Select Feature Columns:")
        self.columns_label.pack()
        self.columns_list = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5)
        self.columns_list.pack(pady=5)
        
        self.target_label = tk.Label(root, text="Target Column (e.g., Sales):")
        self.target_label.pack()
        self.target_var = tk.StringVar()
        self.target_entry = tk.Entry(root, textvariable=self.target_var)
        self.target_entry.pack(pady=5)
        
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)
        
        self.predict_button = tk.Button(root, text="Predict on Loaded Data", command=self.predict)
        self.predict_button.pack(pady=10)
        
        self.trend_button = tk.Button(root, text="Show Sales Trend", command=self.show_trend)
        self.trend_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.columns_list.delete(0, tk.END)
                for col in self.df.columns:
                    self.columns_list.insert(tk.END, col)
                self.result_label.config(text=f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return
        
        selected = self.columns_list.curselection()
        features = [self.columns_list.get(i) for i in selected]
        target = self.target_var.get().strip()
        
        if not features or not target:
            messagebox.showerror("Error", "Please select feature columns and enter a target column.")
            return
        
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Target column '{target}' not found in data.")
            return
        
        try:
            X = self.df[features]
            y = self.df[target]
            
            # Simple preprocessing
            X = X.select_dtypes(include=[float, int])  # Keep only numeric
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            
            self.result_label.config(text=f"Model trained successfully. MSE on test set: {mse:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return
        
        if self.df is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        selected = self.columns_list.curselection()
        features = [self.columns_list.get(i) for i in selected]
        
        try:
            X = self.df[features]
            X = X.select_dtypes(include=[float, int])
            X = X.fillna(X.mean())
            
            predictions = self.model.predict(X)
            self.df['Predicted_Sales'] = predictions
            
            self.result_label.config(text=f"Predictions added to data. Sample: {predictions[:5]}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def show_trend(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        target = self.target_var.get().strip()
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Target column '{target}' not found.")
            return
        
        # Assume there's a 'Date' column
        if 'Date' not in self.df.columns:
            messagebox.showerror("Error", "No 'Date' column found for trend analysis.")
            return
        
        try:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values('Date')
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.df['Date'], self.df[target], marker='o')
            plt.title("Sales Trend Over Time")
            plt.xlabel("Date")
            plt.ylabel(target)
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show trend: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SalesPredictor(root)
    root.mainloop()