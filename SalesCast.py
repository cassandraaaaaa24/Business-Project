import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0F1117"   # near-black canvas
PANEL       = "#181C27"   # card background
BORDER      = "#252A3A"   # subtle border
ACCENT      = "#4F8EF7"   # electric blue
ACCENT2     = "#2DD4BF"   # teal highlight
TEXT        = "#E8EAF0"   # primary text
SUBTEXT     = "#6B7394"   # muted text
SUCCESS     = "#34D399"   # green for success
ERROR_COL   = "#F87171"   # red for errors
HOVER       = "#2A3050"   # hover state

FONT_TITLE  = ("Georgia", 22, "bold")
FONT_HEAD   = ("Georgia", 11, "bold")
FONT_LABEL  = ("Courier", 9)
FONT_MONO   = ("Courier", 9)
FONT_BTN    = ("Georgia", 10, "bold")
FONT_STATUS = ("Courier", 9)


def styled_button(parent, text, command, style="primary", width=20):
    colors = {
        "primary": (ACCENT, BG, TEXT),
        "secondary": (PANEL, BORDER, TEXT),
        "success": (SUCCESS, SUCCESS, BG),
    }
    bg, border_col, fg = colors.get(style, colors["primary"])

    frame = tk.Frame(parent, bg=border_col if style == "secondary" else bg,
                     highlightbackground=border_col, highlightthickness=1, bd=0)

    btn = tk.Button(
        frame, text=text, command=command,
        bg=bg, fg=fg,
        font=FONT_BTN,
        relief="flat", bd=0,
        padx=16, pady=8,
        width=width,
        cursor="hand2",
        activebackground=HOVER if style == "secondary" else ACCENT2,
        activeforeground=TEXT,
    )
    btn.pack(fill="both", expand=True)

    def on_enter(e):
        btn.config(bg=HOVER if style == "secondary" else ACCENT2)
    def on_leave(e):
        btn.config(bg=bg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return frame


def section_card(parent, title=None):
    outer = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                     highlightthickness=1, bd=0)
    if title:
        header = tk.Frame(outer, bg=PANEL)
        header.pack(fill="x", padx=16, pady=(14, 0))
        bar = tk.Frame(header, bg=ACCENT, width=3)
        bar.pack(side="left", fill="y", padx=(0, 8))
        tk.Label(header, text=title.upper(), font=FONT_HEAD,
                 bg=PANEL, fg=TEXT).pack(side="left")
    inner = tk.Frame(outer, bg=PANEL)
    inner.pack(fill="both", expand=True, padx=16, pady=(8, 16))
    return outer, inner


class SalesPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("SALESCAST — Predictive Analytics")
        self.root.configure(bg=BG)
        self.root.geometry("960x700")
        self.root.minsize(860, 600)
        self.df = None
        self.model = None
        self.scaler = None
        self.scale_enabled = tk.BooleanVar(value=False)
        self.model_type = tk.StringVar(value="LinearRegression")
        self.feature_columns = []
        self._build_ui()

    # ── Layout ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        body.columnconfigure(0, weight=1, minsize=260)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)
        self._build_status_bar()

    def _build_header(self):
        bar = tk.Frame(self.root, bg=PANEL, height=60,
                       highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(side="left", fill="y", padx=24)

        dot_frame = tk.Frame(inner, bg=PANEL)
        dot_frame.pack(side="left", anchor="center", pady=12)
        for col in (ACCENT, ACCENT2, SUCCESS):
            tk.Frame(dot_frame, bg=col, width=10, height=10,
                     bd=0).pack(side="left", padx=3)

        tk.Label(inner, text="  SALESCAST", font=FONT_TITLE,
                 bg=PANEL, fg=TEXT).pack(side="left", padx=(12, 0))
        tk.Label(inner, text="  /  Predictive Analytics", font=("Courier", 11),
                 bg=PANEL, fg=SUBTEXT).pack(side="left")

        badge = tk.Label(bar, text="LINEAR REGRESSION",
                         font=("Courier", 8, "bold"),
                         bg=ACCENT, fg=BG, padx=10, pady=4)
        badge.pack(side="right", anchor="center", padx=24)

    def _build_left(self, parent):
        col = tk.Frame(parent, bg=BG)
        col.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)

        # ── Load card ──
        card, inner = section_card(col, "01  Data Source")
        card.pack(fill="x", pady=(0, 10))

        self.file_label = tk.Label(
            inner, text="No file loaded",
            font=FONT_MONO, bg=PANEL, fg=SUBTEXT, wraplength=200, anchor="w")
        self.file_label.pack(fill="x", pady=(0, 8))

        styled_button(inner, "Load CSV File", self.load_csv,
                      style="primary", width=22).pack(fill="x")

        # ── Columns card ──
        card2, inner2 = section_card(col, "02  Feature Columns")
        card2.pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(inner2, text="CTRL+click to select multiple",
                 font=FONT_LABEL, bg=PANEL, fg=SUBTEXT).pack(anchor="w")

        list_frame = tk.Frame(inner2, bg=BORDER, highlightbackground=BORDER,
                              highlightthickness=1, bd=0)
        list_frame.pack(fill="both", expand=True, pady=(6, 0))

        sb = tk.Scrollbar(list_frame, bg=PANEL, troughcolor=PANEL,
                          relief="flat", width=10)
        sb.pack(side="right", fill="y")

        self.columns_list = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            bg=PANEL, fg=TEXT,
            selectbackground=ACCENT, selectforeground=BG,
            font=FONT_MONO,
            relief="flat", bd=4,
            activestyle="none",
            yscrollcommand=sb.set,
            highlightthickness=0,
        )
        self.columns_list.pack(fill="both", expand=True)
        sb.config(command=self.columns_list.yview)

        # ── Target card ──
        card3, inner3 = section_card(col, "03  Target Column")
        card3.pack(fill="x")

        tk.Label(inner3, text="Column name (e.g. Sales)",
                 font=FONT_LABEL, bg=PANEL, fg=SUBTEXT).pack(anchor="w")

        entry_frame = tk.Frame(inner3, bg=BORDER,
                               highlightbackground=ACCENT, highlightthickness=1)
        entry_frame.pack(fill="x", pady=(6, 0))

        self.target_var = tk.StringVar()
        self.target_entry = tk.Entry(
            entry_frame,
            textvariable=self.target_var,
            bg=PANEL, fg=TEXT, insertbackground=ACCENT,
            font=FONT_MONO, relief="flat", bd=8,
        )
        self.target_entry.pack(fill="x")

        # ── Scaling option ──
        card4, inner4 = section_card(col, "04  Preprocessing")
        card4.pack(fill="x", pady=(10, 0))

        scale_check = tk.Checkbutton(
            inner4, text="Apply Feature Scaling (StandardScaler)",
            variable=self.scale_enabled,
            bg=PANEL, fg=TEXT, selectcolor=ACCENT, activebackground=PANEL,
            activeforeground=TEXT, font=FONT_LABEL, cursor="hand2"
        )
        scale_check.pack(anchor="w", pady=4)

        # ── Model selection ──
        card5, inner5 = section_card(col, "05  Algorithm")
        card5.pack(fill="x", pady=(10, 0))

        tk.Label(inner5, text="Select regression model:",
                 font=FONT_LABEL, bg=PANEL, fg=SUBTEXT).pack(anchor="w", pady=(0, 6))

        model_frame = tk.Frame(inner5, bg=BORDER,
                               highlightbackground=ACCENT, highlightthickness=1)
        model_frame.pack(fill="x")

        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_type,
            values=["LinearRegression", "Ridge", "Lasso", "RandomForest", "SVR"],
            state="readonly",
            font=FONT_MONO,
            width=20
        )
        self.model_combo.pack(fill="x", padx=1, pady=1)
        # Style combobox
        self.model_combo.configure(foreground=TEXT)
        style = ttk.Style()
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL)


    def _build_right(self, parent):
        col = tk.Frame(parent, bg=BG)
        col.grid(row=0, column=1, sticky="nsew", pady=12)
        col.rowconfigure(1, weight=1)

        # ── Action buttons ──
        btn_card, btn_inner = section_card(col, "Actions")
        btn_card.pack(fill="x", pady=(0, 10))

        btn_row = tk.Frame(btn_inner, bg=PANEL)
        btn_row.pack(fill="x")
        btn_row.columnconfigure((0, 1, 2, 3), weight=1)

        styled_button(btn_inner, "▶  Train Model", self.train_model,
                      style="primary").pack(side="left", padx=(0, 8), expand=True, fill="x")
        styled_button(btn_inner, "◆  Run Predictions", self.predict,
                      style="secondary").pack(side="left", padx=(0, 8), expand=True, fill="x")
        styled_button(btn_inner, "⟜  Sales Trend", self.show_trend,
                      style="secondary").pack(side="left", padx=(0, 8), expand=True, fill="x")
        styled_button(btn_inner, "💾  Export CSV", self.export_predictions,
                      style="secondary").pack(side="left", padx=(0, 8), expand=True, fill="x")
        styled_button(btn_inner, "✕  Exit", self.exit_app,
                      style="secondary").pack(side="left", expand=True, fill="x")

        # ── Custom prediction card ──
        card_custom, inner_custom = section_card(col, "Custom Prediction")
        card_custom.pack(fill="x", pady=(0, 10))

        tk.Label(inner_custom, text="Input values for quick prediction:",
                 font=FONT_LABEL, bg=PANEL, fg=SUBTEXT).pack(anchor="w", pady=(0, 6))

        self.custom_inputs_frame = tk.Frame(inner_custom, bg=PANEL)
        self.custom_inputs_frame.pack(fill="x")

        styled_button(inner_custom, "📊  Predict", self.predict_custom,
                      style="success").pack(fill="x", pady=(6, 0))

        # ── Results / plot area ──
        card, inner = section_card(col, "Output")
        card.pack(fill="both", expand=True)

        self.output_frame = inner
        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.output_frame.winfo_children():
            w.destroy()
        ph = tk.Frame(self.output_frame, bg=PANEL)
        ph.pack(fill="both", expand=True)
        tk.Label(ph, text="⬡", font=("Georgia", 48), bg=PANEL,
                 fg=BORDER).pack(expand=True)
        tk.Label(ph, text="Load data & train a model to see results",
                 font=FONT_LABEL, bg=PANEL, fg=SUBTEXT).pack()

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=PANEL, height=30,
                       highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_dot = tk.Label(bar, text="●", font=("Courier", 10),
                                   bg=PANEL, fg=SUBTEXT)
        self.status_dot.pack(side="left", padx=(14, 4))

        self.status_label = tk.Label(bar, text="Ready",
                                     font=FONT_STATUS, bg=PANEL, fg=SUBTEXT, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)

        tk.Label(bar, text="sklearn  ·  pandas  ·  matplotlib",
                 font=FONT_STATUS, bg=PANEL, fg=BORDER).pack(side="right", padx=14)

    # ── Status helpers ──────────────────────────────────────────────────────────
    def _set_status(self, msg, kind="info"):
        colors = {"info": SUBTEXT, "success": SUCCESS, "error": ERROR_COL}
        dots   = {"info": SUBTEXT, "success": SUCCESS, "error": ERROR_COL}
        self.status_label.config(text=msg, fg=colors.get(kind, SUBTEXT))
        self.status_dot.config(fg=dots.get(kind, SUBTEXT))

    def _show_result_text(self, lines):
        """Render structured text results in the output panel."""
        for w in self.output_frame.winfo_children():
            w.destroy()

        canvas = tk.Canvas(self.output_frame, bg=PANEL, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        y = 20
        for kind, text in lines:
            if kind == "metric":
                key, val = text
                canvas.create_text(20, y, text=key, anchor="nw",
                                   font=FONT_LABEL, fill=SUBTEXT)
                canvas.create_text(340, y, text=val, anchor="ne",
                                   font=("Courier", 9, "bold"), fill=ACCENT2)
                y += 22
            elif kind == "divider":
                canvas.create_line(20, y + 4, 340, y + 4, fill=BORDER)
                y += 16
            elif kind == "title":
                canvas.create_text(20, y, text=text, anchor="nw",
                                   font=FONT_HEAD, fill=TEXT)
                y += 28
            elif kind == "sample":
                canvas.create_text(20, y, text=text, anchor="nw",
                                   font=FONT_MONO, fill=SUBTEXT)
                y += 18

    # ── Core logic ──────────────────────────────────────────────────────────────
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            self.columns_list.delete(0, tk.END)
            for col in self.df.columns:
                self.columns_list.insert(tk.END, f"  {col}")
            short = path.split("/")[-1]
            self.file_label.config(text=f"✔  {short}", fg=SUCCESS)
            self._set_status(
                f"Loaded '{short}' — {len(self.df):,} rows · {len(self.df.columns)} columns",
                "success")
            self.show_data_summary()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._set_status("Failed to load file", "error")

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        selected = self.columns_list.curselection()
        features = [self.columns_list.get(i).strip() for i in selected]
        target   = self.target_var.get().strip()

        if not features or not target:
            messagebox.showerror("Error", "Select feature columns and enter a target column.")
            return
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Column '{target}' not found.")
            return

        try:
            X = self.df[features].select_dtypes(include=[float, int])
            X = X.fillna(X.mean())
            y = self.df[target].fillna(self.df[target].mean())

            # Store feature columns for custom predictions
            self.feature_columns = X.columns.tolist()
            self._build_custom_inputs()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Apply scaling if enabled
            if self.scale_enabled.get():
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            else:
                self.scaler = None

            # Create model based on selection
            model_name = self.model_type.get()
            if model_name == "LinearRegression":
                self.model = LinearRegression()
            elif model_name == "Ridge":
                self.model = Ridge(alpha=1.0)
            elif model_name == "Lasso":
                self.model = Lasso(alpha=0.1)
            elif model_name == "RandomForest":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "SVR":
                self.model = SVR(kernel='rbf', C=100)
            else:
                self.model = LinearRegression()

            self.model.fit(X_train, y_train)

            pred  = self.model.predict(X_test)
            mse   = mean_squared_error(y_test, pred)
            rmse  = mse ** 0.5
            mae   = mean_absolute_error(y_test, pred)
            score = self.model.score(X_test, y_test)

            lines = [
                ("title", "Training Results"),
                ("divider", None),
                ("metric", ("Model",           model_name)),
                ("metric", ("Features used",     str(len(X.columns)))),
                ("metric", ("Training samples",  f"{len(X_train):,}")),
                ("metric", ("Test samples",       f"{len(X_test):,}")),
                ("divider", None),
                ("metric", ("MAE",  f"{mae:,.4f}")),
                ("metric", ("MSE",   f"{mse:,.4f}")),
                ("metric", ("RMSE",  f"{rmse:,.4f}")),
                ("metric", ("R²",    f"{score:.4f}")),
                ("divider", None),
                ("metric", ("Scaling", "Enabled" if self.scaler else "Disabled")),
            ]
            self._show_result_text(lines)
            self._set_status(f"{model_name} trained  ·  R² = {score:.4f}  ·  RMSE = {rmse:,.2f}", "success")

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self._set_status("Training failed", "error")

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Train the model first.")
            return
        if self.df is None:
            messagebox.showerror("Error", "Load data first.")
            return

        selected = self.columns_list.curselection()
        features = [self.columns_list.get(i).strip() for i in selected]

        try:
            X = self.df[features].select_dtypes(include=[float, int])
            X = X.fillna(X.mean())
            
            # Apply scaling if it was used during training
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            preds = self.model.predict(X)
            self.df["Predicted_Sales"] = preds

            samples = [f"  {v:>10.2f}" for v in preds[:8]]
            lines = [
                ("title", "Predictions"),
                ("divider", None),
                ("metric", ("Rows predicted", f"{len(preds):,}")),
                ("metric", ("Min",  f"{preds.min():,.2f}")),
                ("metric", ("Max",  f"{preds.max():,.2f}")),
                ("metric", ("Mean", f"{preds.mean():,.2f}")),
                ("divider", None),
                ("title", "Sample (first 8)"),
            ] + [("sample", s) for s in samples]

            self._show_result_text(lines)
            self._set_status(f"Predictions generated for {len(preds):,} rows", "success")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._set_status("Prediction failed", "error")

    def exit_app(self):
        self.root.quit()
        self.root.destroy()

    def _build_custom_inputs(self):
        """Build input fields for custom predictions."""
        for w in self.custom_inputs_frame.winfo_children():
            w.destroy()
        
        self.custom_vars = {}
        for col in self.feature_columns:
            row = tk.Frame(self.custom_inputs_frame, bg=PANEL)
            row.pack(fill="x", pady=2)
            
            tk.Label(row, text=col[:12], font=FONT_LABEL, bg=PANEL, fg=SUBTEXT,
                     width=12, anchor="w").pack(side="left", padx=(0, 4))
            
            var = tk.StringVar()
            self.custom_vars[col] = var
            
            entry = tk.Entry(row, textvariable=var, bg=BORDER, fg=TEXT,
                           insertbackground=ACCENT, font=FONT_MONO, relief="flat",
                           width=12, bd=2)
            entry.pack(side="left", fill="x", expand=True)

    def predict_custom(self):
        """Make a single prediction from custom input values."""
        if self.model is None:
            messagebox.showerror("Error", "Train the model first.")
            return
        
        if not self.feature_columns:
            messagebox.showerror("Error", "No features available.")
            return
        
        try:
            # Gather input values
            input_values = []
            for col in self.feature_columns:
                val = self.custom_vars[col].get().strip()
                if not val:
                    messagebox.showerror("Error", f"Please enter a value for '{col}'.")
                    return
                input_values.append(float(val))
            
            # Convert to array
            X_custom = np.array([input_values])
            
            # Apply scaling if it was used during training
            if self.scaler is not None:
                X_custom = self.scaler.transform(X_custom)
            
            pred = self.model.predict(X_custom)[0]
            
            lines = [
                ("title", "Single Prediction"),
                ("divider", None),
            ]
            
            # Show input values
            for col, val in zip(self.feature_columns, input_values):
                lines.append(("metric", (f"  {col}", f"{val:,.2f}")))
            
            lines.append(("divider", None))
            lines.append(("metric", ("Predicted Value", f"{pred:,.2f}")))
            
            self._show_result_text(lines)
            self._set_status(f"Prediction: {pred:,.2f}", "success")
        
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
            self._set_status("Custom prediction failed", "error")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._set_status("Custom prediction failed", "error")


    def show_data_summary(self):
        """Display summary statistics of loaded data."""
        if self.df is None:
            return
        
        numeric_cols = self.df.select_dtypes(include=[float, int]).columns
        lines = [("title", "Data Summary"), ("divider", None)]
        
        lines.append(("metric", ("Total Rows", f"{len(self.df):,}")))
        lines.append(("metric", ("Total Columns", str(len(self.df.columns)))))
        
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            lines.append(("metric", ("Missing Values", str(missing))))
        
        lines.append(("divider", None))
        lines.append(("title", "Numeric Columns"))
        
        for col in numeric_cols[:5]:  # Show first 5 numeric columns
            lines.append(("metric", (f"  {col} Mean", f"{self.df[col].mean():.2f}")))
            lines.append(("metric", (f"  {col} Std Dev", f"{self.df[col].std():.2f}")))
        
        if len(numeric_cols) > 5:
            lines.append(("sample", f"  ... and {len(numeric_cols) - 5} more columns"))
        
        self._show_result_text(lines)

    def export_predictions(self):
        """Export dataframe with predictions to CSV."""
        if self.df is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        
        if "Predicted_Sales" not in self.df.columns:
            messagebox.showerror("Error", "Run predictions first to export.")
            return
        
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if path:
                self.df.to_csv(path, index=False)
                self._set_status(f"Exported to {path.split('/')[-1]}", "success")
                messagebox.showinfo("Success", f"Predictions exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self._set_status("Export failed", "error")

    def show_trend(self):
        if self.df is None:
            messagebox.showerror("Error", "Load data first.")
            return

        target = self.target_var.get().strip()
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Target column '{target}' not found.")
            return
        if "Date" not in self.df.columns:
            messagebox.showerror("Error", "No 'Date' column found for trend analysis.")
            return

        try:
            df = self.df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

            # ── Matplotlib styled to match app theme ──
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor(PANEL)
            ax.set_facecolor(BG)

            ax.plot(df["Date"], df[target],
                    color=ACCENT, linewidth=1.8, zorder=3)
            ax.fill_between(df["Date"], df[target],
                            alpha=0.15, color=ACCENT)

            if "Predicted_Sales" in df.columns:
                ax.plot(df["Date"], df["Predicted_Sales"],
                        color=ACCENT2, linewidth=1.4,
                        linestyle="--", label="Predicted", zorder=4)
                ax.legend(facecolor=PANEL, edgecolor=BORDER,
                          labelcolor=TEXT, fontsize=9)

            ax.set_title("Sales Trend Over Time", color=TEXT,
                         fontsize=13, fontfamily="serif", pad=14)
            ax.set_xlabel("Date", color=SUBTEXT, fontsize=9)
            ax.set_ylabel(target, color=SUBTEXT, fontsize=9)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
            ax.spines[:].set_color(BORDER)
            ax.grid(color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
            fig.tight_layout()

            # Embed in output panel
            for w in self.output_frame.winfo_children():
                w.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            self._set_status("Sales trend chart rendered", "success")

        except Exception as e:
            messagebox.showerror("Trend Error", str(e))
            self._set_status("Failed to render trend", "error")


if __name__ == "__main__":
    root = tk.Tk()
    app = SalesPredictor(root)
    root.mainloop()