
import tkinter as tk
from tkinter import ttk, messagebox
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. Enable Crisp High-DPI Scaling for Windows (prevents blurry text)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Matplotlib global typography & aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "Segoe UI",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "figure.dpi": 100
})

def brighten_color(hex_str, factor=1.20):
    """Brightens hex color for hover effect without drawing unwanted lines."""
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"

class PillButton(tk.Canvas):
    """Seamless capsule button with generous width preventing text truncation."""
    def __init__(self, parent, text, bg_color, command=None, width=335, height=52, text_color="#FFFFFF", align="left"):
        super().__init__(parent, width=width, height=height, bg="#0F172A", highlightthickness=0, cursor="hand2")
        self.command = command
        self.base_color = bg_color
        self.hover_color = brighten_color(bg_color)
        self.text_str = text
        self.text_color = text_color
        self.align = align
        self.w = width
        self.h = height

        self.draw_pill(self.base_color)

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def draw_pill(self, fill_color):
        self.delete("all")
        x1, y1, x2, y2 = 4, 3, self.w - 4, self.h - 3
        r = (y2 - y1) // 2

        # Draw seamless capsule using smooth end ovals and middle rectangle (No dividing lines)
        self.create_oval(x1, y1, x1 + 2*r, y2, fill=fill_color, outline=fill_color)
        self.create_oval(x2 - 2*r, y1, x2, y2, fill=fill_color, outline=fill_color)
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill_color, outline=fill_color)

        # Left alignment with 22px margin leaves over 75px buffer on right so 8 & 9 never cut off
        if self.align == "center":
            self.create_text(self.w // 2, self.h // 2, text=self.text_str, fill=self.text_color, 
                             font=("Segoe UI", 9, "bold"), anchor="center")
        else:
            self.create_text(22, self.h // 2, text=self.text_str, fill=self.text_color, 
                             font=("Segoe UI", 9, "bold"), anchor="w")

    def on_enter(self, e):
        self.draw_pill(self.hover_color)

    def on_leave(self, e):
        self.draw_pill(self.base_color)

    def on_click(self, e):
        if self.command:
            self.command()

class AIWorkplaceExecutiveSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI & Future of Work in India | Executive Intelligence Suite")
        self.geometry("1560x940")
        self.minsize(1360, 820)
        self.configure(bg="#0B132B")

        # 1. Load Data & Train ML Model
        self.load_dataset()
        self.train_ml_model()

        # 2. Build UI Hierarchy
        self.create_top_kpi_bar()
        self.create_main_workspace()
        self.show_home_view()

    def load_dataset(self):
        try:
            self.df = pd.read_csv("fact_ai_workplace_raw.csv")
        except Exception:
            try:
                self.df = pd.read_excel("MainDatasetEXL.xlsx")
            except Exception as e:
                messagebox.showerror("Data Error", f"Cannot load dataset: {e}")
                self.df = pd.DataFrame()
                return

        # Numeric transformations
        self.df["Wage_Num"] = (
            self.df["Avg_Monthly_Wage_INR"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        self.df["Wage_Num"] = pd.to_numeric(self.df["Wage_Num"], errors="coerce")

        for col in ["AI_Exposure_Score", "Automation_Potential_Pct", "Augmentation_Potential_Pct", 
                    "Estimated_Workers_Thousands", "Emerging_Skill_Importance_Score"]:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
                )
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Longitudinal Summary Table
        self.occ_summary = self.df.pivot_table(
            index=["NCO_Code", "Occupation_Title", "Industry_Sector", "AI_Exposure_Score", 
                   "Exposure_Tier", "Automation_Potential_Pct", "Augmentation_Potential_Pct"],
            columns="Year",
            values=["Estimated_Workers_Thousands", "Wage_Num"],
            aggfunc={"Estimated_Workers_Thousands": "sum", "Wage_Num": "mean"}
        ).reset_index()
        self.occ_summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in self.occ_summary.columns]

        self.occ_summary["Net_Growth_Pct"] = (
            (self.occ_summary["Estimated_Workers_Thousands_2023"] - self.occ_summary["Estimated_Workers_Thousands_2019"])
            / self.occ_summary["Estimated_Workers_Thousands_2019"]
        ) * 100

        self.occ_summary["Wage_Growth_Pct"] = (
            (self.occ_summary["Wage_Num_2023"] - self.occ_summary["Wage_Num_2019"])
            / self.occ_summary["Wage_Num_2019"]
        ) * 100

        # Build occupational time-series table for fast historical lookup
        self.occ_ts_data = self.df.pivot_table(
            index="Occupation_Title",
            columns="Year",
            values="Estimated_Workers_Thousands",
            aggfunc="sum"
        )

    def train_ml_model(self):
        features = ["AI_Exposure_Score", "Automation_Potential_Pct", "Augmentation_Potential_Pct"]
        X = self.occ_summary[features].fillna(0)
        y = self.occ_summary["Net_Growth_Pct"].fillna(0)

        self.ml_model = LinearRegression()
        self.ml_model.fit(X, y)
        self.r2_metric = self.ml_model.score(X, y)

    def refresh_data_and_model(self):
        self.load_dataset()
        self.train_ml_model()
        messagebox.showinfo("Data Updated", "Dataset reloaded and ML Forecasting Model re-trained successfully!")
        self.show_home_view()

    def create_top_kpi_bar(self):
        kpi_container = tk.Frame(self, bg="#FFFFFF", height=78, relief="solid", bd=1)
        kpi_container.pack(side="top", fill="x", padx=14, pady=(10, 6))

        kpis = [
            ("TOTAL WORKFORCE (2023)", f"{self.df[self.df['Year']==2023]['Estimated_Workers_Thousands'].sum():,.1f}k", "#0F172A"),
            ("AVG MONTHLY WAGE (2023)", f"₹{self.df[self.df['Year']==2023]['Wage_Num'].mean():,.0f}", "#0D9488"),
            ("MEAN AI EXPOSURE SCORE", f"{self.df['AI_Exposure_Score'].mean():.2f} / 1.0", "#2563EB"),
            ("ML MODEL ACCURACY (R²)", f"{self.r2_metric:.2%}", "#059669")
        ]

        for label, val, highlight_color in kpis:
            card = tk.Frame(kpi_container, bg="#FFFFFF", bd=1, relief="solid")
            card.pack(side="left", fill="both", expand=True, padx=6, pady=6)

            indicator = tk.Frame(card, bg=highlight_color, width=4)
            indicator.pack(side="left", fill="y")

            body = tk.Frame(card, bg="#FFFFFF")
            body.pack(side="left", fill="both", expand=True, padx=12, pady=4)
            tk.Label(body, text=label, font=("Segoe UI", 8, "bold"), fg="#64748B", bg="#FFFFFF").pack(anchor="w")
            tk.Label(body, text=val, font=("Segoe UI", 14, "bold"), fg=highlight_color, bg="#FFFFFF").pack(anchor="w", pady=(2, 0))

    def create_main_workspace(self):
        workspace = tk.Frame(self, bg="#0B132B")
        workspace.pack(side="bottom", fill="both", expand=True, padx=14, pady=(0, 10))

        # LEFT SIDEBAR: Width 365 to fit 335px buttons comfortably
        self.sidebar = tk.Frame(workspace, bg="#0F172A", width=365, relief="solid", bd=1)
        self.sidebar.pack(side="left", fill="y", padx=(0, 12), pady=0)
        self.sidebar.pack_propagate(False)

        # Header Pill: Executive Controls
        PillButton(self.sidebar, "EXECUTIVE CONTROLS", "#1E293B", self.show_home_view, 
                   width=335, height=44, text_color="#34D399", align="center").pack(pady=(10, 4))

        # Header Pill: Executive Overview (Home)
        PillButton(self.sidebar, "Executive Overview (Home)", "#334155", self.show_home_view, 
                   width=335, height=40, text_color="#F8FAFC", align="center").pack(pady=(0, 6))

        # Separator Line
        sep = tk.Frame(self.sidebar, bg="#334155", height=1)
        sep.pack(fill="x", padx=16, pady=(0, 6))

        # Container for 9 Buttons - Height 52px each spans the full vertical length
        buttons_container = tk.Frame(self.sidebar, bg="#0F172A")
        buttons_container.pack(fill="both", expand=True)

        green_shades_definitions = [
            ("1. Macro Exposure Tiers", "#34D399", self.view_macro_tiers),       # Light Mint / Vibrant Emerald
            ("2. Task Divergence Model", "#2EC28D", self.view_task_divergence),   # Soft Mint Green
            ("3. Net Employment Shifts", "#28B282", self.view_employment_shifts), # Sea Green
            ("4. AI vs Growth Correlation", "#23A176", self.view_exposure_vs_growth), # Medium Emerald
            ("5. Longitudinal Wage Shifts", "#1D906A", self.view_wage_trajectories), # Rich Green
            ("6. Emerging Skills Ranking", "#17805E", self.view_emerging_skills), # Classic Forest Green
            ("7. Industry Sensitivity", "#126F52", self.view_industry_analysis),   # Deep Forest
            ("8. Statistical Correlation Matrix", "#0C5F47", self.view_correlation_matrix), # Pine Green
            ("9. ML Predictive Simulator (2027)", "#064E3B", self.view_ml_simulator) # Deepest Pine / Forest Green
        ]

        for text, color_hex, action in green_shades_definitions:
            PillButton(buttons_container, text, color_hex, action, width=335, height=52, align="left").pack(pady=6)

        # Refresh dataset button at absolute bottom
        PillButton(self.sidebar, "Reload & Refresh Dataset", "#1E293B", self.refresh_data_and_model, 
                   width=335, height=42, text_color="#94A3B8", align="center").pack(side="bottom", pady=(4, 10))

        # RIGHT CANVAS PANEL
        self.canvas_panel = tk.Frame(workspace, bg="#FFFFFF", relief="solid", bd=1)
        self.canvas_panel.pack(side="right", fill="both", expand=True)

    def clear_panel(self):
        for widget in self.canvas_panel.winfo_children():
            widget.destroy()

    def build_header_bar(self, view_title, has_drill_filter=False):
        header = tk.Frame(self.canvas_panel, bg="#F8FAFC", height=50, bd=1, relief="groove")
        header.pack(side="top", fill="x", padx=14, pady=(10, 6))

        tk.Label(header, text=view_title, font=("Segoe UI", 11, "bold"), 
                 fg="#0F172A", bg="#F8FAFC").pack(side="left", padx=12, pady=8)

        back_btn = tk.Button(
            header, text="Return to Overview", font=("Segoe UI", 9, "bold"),
            bg="#1E293B", fg="#FFFFFF", activebackground="#334155", activeforeground="#FFFFFF",
            bd=0, padx=12, pady=5, cursor="hand2", command=self.show_home_view
        )
        back_btn.pack(side="right", padx=12, pady=6)

        if has_drill_filter:
            filter_container = tk.Frame(self.canvas_panel, bg="#FFFFFF")
            filter_container.pack(side="top", fill="x", padx=16, pady=(0, 6))

            tk.Label(filter_container, text="Drill Through (Exposure Tier):", font=("Segoe UI", 9, "bold"), 
                     fg="#334155", bg="#FFFFFF").pack(side="left", padx=4)

            self.tier_filter_var = tk.StringVar(value="All")
            combo = ttk.Combobox(
                filter_container, textvariable=self.tier_filter_var, 
                values=["All", "High", "Moderate", "Low"], state="readonly", width=14
            )
            combo.pack(side="left", padx=8)
            combo.bind("<<ComboboxSelected>>", lambda e: self.apply_tier_filter())

    def apply_tier_filter(self):
        choice = self.tier_filter_var.get()
        data = self.occ_summary if choice == "All" else self.occ_summary[self.occ_summary["Exposure_Tier"] == choice]
        self.draw_task_divergence_chart(data)

    def render_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(4, 14))
        plt.close(fig)

    # ==================== VIEW 0: HOME EXECUTIVE OVERVIEW ====================
    def show_home_view(self):
        self.clear_panel()

        container = tk.Frame(self.canvas_panel, bg="#FFFFFF")
        container.pack(fill="both", expand=True, padx=25, pady=20)

        tk.Label(container, text="AI AND THE FUTURE OF WORK IN INDIA", 
                 font=("Segoe UI", 20, "bold"), fg="#0F172A", bg="#FFFFFF", anchor="w").pack(fill="x", pady=(0, 2))
        tk.Label(container, text="Longitudinal Empirical Labor Study (2019–2023) | Microdata NCO-NIC Framework",
                 font=("Segoe UI", 11, "bold"), fg="#059669", bg="#FFFFFF", anchor="w").pack(fill="x", pady=(0, 14))

        abstract_card = tk.Frame(container, bg="#F8FAFC", bd=1, relief="solid")
        abstract_card.pack(fill="x", pady=(0, 18), ipady=8, ipadx=12)

        p1_title = "Hypothesis & Scope:"
        p1_body = (
            "This project analyzes the structural impact of Artificial Intelligence and Generative AI across the Indian "
            "labor ecosystem using 2,058 standardized occupational-regional panel records. Rather than demonstrating "
            "blanket workforce contraction, the empirical data validates a Task-Level Divergence Model."
        )
        p2_title = "Empirical Highlights & Verification:"
        p2_body = (
            "• High AI Exposure (+9.08%) & Moderate Exposure (+11.68%) maintained net workforce expansion nationally.\n"
            "• Polarization Gap: Routine clerical positions (Data Entry, Bank Tellers) contracted by -15.5%, whereas "
            "cognitively augmented roles (Software Engineering, Systems Architecture) expanded by +33.6%.\n"
            "• Machine Learning Integration: Dynamic forecasting engine models occupation trajectories up to 2027 "
            "with interactive scenario updates."
        )

        tk.Label(abstract_card, text=p1_title, font=("Segoe UI", 9, "bold"), fg="#0F172A", bg="#F8FAFC", anchor="w").pack(fill="x", padx=16, pady=(6, 2))
        tk.Label(abstract_card, text=p1_body, font=("Segoe UI", 9), fg="#334155", bg="#F8FAFC", justify="left", wraplength=950, anchor="w").pack(fill="x", padx=16, pady=(0, 8))

        tk.Label(abstract_card, text=p2_title, font=("Segoe UI", 9, "bold"), fg="#0F172A", bg="#F8FAFC", anchor="w").pack(fill="x", padx=16, pady=(4, 2))
        tk.Label(abstract_card, text=p2_body, font=("Segoe UI", 9), fg="#475569", bg="#F8FAFC", justify="left", wraplength=950, anchor="w").pack(fill="x", padx=16, pady=(0, 6))

        grid_frame = tk.Frame(container, bg="#FFFFFF")
        grid_frame.pack(fill="x", pady=6)

        cards = [
            ("Moderate Exposure Growth", "+11.68%", "Resilient operational & tech-enabled roles", "#10B981"),
            ("High Exposure Growth", "+9.08%", "Masked polarization along task lines", "#059669"),
            ("Low Exposure Growth", "+5.39%", "Field & physical infrastructure roles", "#D97706"),
            ("Structural Polarization Gap", "49.16%", "Software Dev (+33.6%) vs Typists (-15.5%)", "#064E3B")
        ]

        for i, (title, stat, desc, col) in enumerate(cards):
            c = tk.Frame(grid_frame, bg="#FFFFFF", bd=1, relief="solid", padx=14, pady=12)
            c.grid(row=0, column=i, padx=5, sticky="nsew")
            grid_frame.grid_columnconfigure(i, weight=1)

            tk.Label(c, text=title, font=("Segoe UI", 8, "bold"), fg="#64748B", bg="#FFFFFF", anchor="w").pack(fill="x")
            tk.Label(c, text=stat, font=("Segoe UI", 16, "bold"), fg=col, bg="#FFFFFF", anchor="w").pack(fill="x", pady=2)
            tk.Label(c, text=desc, font=("Segoe UI", 8), fg="#475569", bg="#FFFFFF", anchor="w")

    # ==================== VIEW 1: MACRO TIERS ====================
    def view_macro_tiers(self):
        self.clear_panel()
        self.build_header_bar("Macro Employment Trajectory by AI Exposure Tier (2019–2023)")

        tier_growth = self.df.pivot_table(
            index="Exposure_Tier", columns="Year", values="Estimated_Workers_Thousands", aggfunc="sum"
        ).loc[["High", "Moderate", "Low"]]

        fig, ax = plt.subplots(figsize=(10.5, 5))
        tier_growth.plot(kind="bar", ax=ax, color=["#A7F3D0", "#34D399", "#059669"], width=0.65, edgecolor="none")
        ax.set_title("Longitudinal Estimated Workforce (Thousands) by Year & Tier", fontweight="bold", pad=12)
        ax.set_xlabel("AI Exposure Tier", fontweight="bold", labelpad=8)
        ax.set_ylabel("Workers (in Thousands)", fontweight="bold", labelpad=8)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(title="Year", frameon=True)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.0fk", padding=3, fontsize=8, fontweight="bold")

        fig.tight_layout()
        self.render_plot(fig)

    # ==================== VIEW 2: TASK DIVERGENCE ====================
    def view_task_divergence(self):
        self.clear_panel()
        self.build_header_bar("Task Polarization: Automation Potential vs. Augmentation Potential", has_drill_filter=True)
        self.draw_task_divergence_chart(self.occ_summary)

    def draw_task_divergence_chart(self, dataset):
        for widget in self.canvas_panel.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()

        sample = dataset.sort_values(by="Automation_Potential_Pct", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        fig.subplots_adjust(left=0.34, right=0.96, top=0.92, bottom=0.12)

        bars_auto = ax.barh(sample["Occupation_Title"], sample["Automation_Potential_Pct"], label="Automation Potential %", color="#EF4444", height=0.65)
        bars_aug = ax.barh(sample["Occupation_Title"], sample["Augmentation_Potential_Pct"], left=sample["Automation_Potential_Pct"],
                            label="Augmentation Potential %", color="#10B981", height=0.65)

        ax.set_title("Occupational Task Composition Breakdown (%)", fontweight="bold", pad=10)
        ax.set_xlabel("Task Potential (%)", fontweight="bold")
        ax.set_xlim(0, 105)
        ax.legend(loc="lower right", frameon=True)
        ax.invert_yaxis()

        for bar, val in zip(bars_auto, sample["Automation_Potential_Pct"]):
            if val > 15:
                ax.text(bar.get_x() + val/2, bar.get_y() + bar.get_height()/2, f"{int(val)}%", 
                        ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        for bar, left, val in zip(bars_aug, sample["Automation_Potential_Pct"], sample["Augmentation_Potential_Pct"]):
            if val > 15:
                ax.text(left + val/2, bar.get_y() + bar.get_height()/2, f"{int(val)}%", 
                        ha="center", va="center", color="white", fontweight="bold", fontsize=8)

        self.render_plot(fig)

    # ==================== VIEW 3: NET EMPLOYMENT SHIFTS ====================
    def view_employment_shifts(self):
        self.clear_panel()
        self.build_header_bar("Net Headcount Divergence: Fastest Growing vs Contracting Occupations")

        top_expand = self.occ_summary.sort_values(by="Net_Growth_Pct", ascending=False).head(6)
        top_contract = self.occ_summary.sort_values(by="Net_Growth_Pct", ascending=True).head(6)
        combined = pd.concat([top_expand, top_contract]).sort_values(by="Net_Growth_Pct")

        bar_colors = ["#EF4444" if x < 0 else "#10B981" for x in combined["Net_Growth_Pct"]]

        fig, ax = plt.subplots(figsize=(11, 5.2))
        fig.subplots_adjust(left=0.36, right=0.92, top=0.90, bottom=0.12)

        bars = ax.barh(combined["Occupation_Title"], combined["Net_Growth_Pct"], color=bar_colors, height=0.62)
        ax.axvline(0, color="#475569", linewidth=1.2, linestyle="--")
        ax.set_title("Net Growth Polarization Gap (2019 to 2023 % Shift)", fontweight="bold", pad=10)
        ax.set_xlabel("Net Employment Growth (%)", fontweight="bold")
        ax.set_xlim(-24, 42)

        for bar, val in zip(bars, combined["Net_Growth_Pct"]):
            if val >= 0:
                ax.text(val + 0.8, bar.get_y() + bar.get_height()/2, f"+{val:.1f}%", 
                        ha="left", va="center", fontweight="bold", fontsize=8.5, color="#059669")
            else:
                ax.text(val - 0.8, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", 
                        ha="right", va="center", fontweight="bold", fontsize=8.5, color="#DC2626")

        self.render_plot(fig)

    # ==================== VIEW 4: CLEAN SCATTER PLOT ====================
    def view_exposure_vs_growth(self):
        self.clear_panel()
        self.build_header_bar("Core Thesis Test: AI Exposure Score vs. Net Growth %")

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        fig.subplots_adjust(left=0.10, right=0.80, top=0.90, bottom=0.14)

        tier_colors = {"High": "#EF4444", "Moderate": "#F59E0B", "Low": "#10B981"}
        ax.scatter(
            self.occ_summary["AI_Exposure_Score"],
            self.occ_summary["Net_Growth_Pct"],
            c=self.occ_summary["Exposure_Tier"].map(tier_colors),
            s=self.occ_summary["Estimated_Workers_Thousands_2023"] / 12,
            alpha=0.75,
            edgecolors="none"
        )

        sns.regplot(data=self.occ_summary, x="AI_Exposure_Score", y="Net_Growth_Pct", scatter=False, ax=ax, color="#334155")

        top_role = self.occ_summary.loc[self.occ_summary["Net_Growth_Pct"].idxmax()]
        ax.annotate(
            f"{top_role['Occupation_Title']}\n(+{top_role['Net_Growth_Pct']:.1f}% growth | Aug {int(top_role['Augmentation_Potential_Pct'])}%)",
            xy=(top_role["AI_Exposure_Score"], top_role["Net_Growth_Pct"]),
            xytext=(-170, 16), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#059669",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#059669", lw=1.5),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2)
        )

        bot_role = self.occ_summary.loc[self.occ_summary["Net_Growth_Pct"].idxmin()]
        ax.annotate(
            f"{bot_role['Occupation_Title']}\n({bot_role['Net_Growth_Pct']:.1f}% contraction | Auto {int(bot_role['Automation_Potential_Pct'])}%)",
            xy=(bot_role["AI_Exposure_Score"], bot_role["Net_Growth_Pct"]),
            xytext=(-175, -25), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#DC2626",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#DC2626", lw=1.5),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2)
        )

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='High Exposure', markerfacecolor='#EF4444', markersize=9),
            plt.Line2D([0], [0], marker='o', color='w', label='Moderate Exposure', markerfacecolor='#F59E0B', markersize=9),
            plt.Line2D([0], [0], marker='o', color='w', label='Low Exposure', markerfacecolor='#10B981', markersize=9),
            plt.Line2D([0], [0], color='#334155', lw=2, label='Regression Trend')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True, title="Exposure Tier", fontsize=8.5)

        ax.set_title("Empirical Divergence: High Exposure Yields Both Expansion & Contraction", fontweight="bold", pad=12)
        ax.set_xlabel("AI Exposure Score (0.0 to 1.0)", fontweight="bold", labelpad=8)
        ax.set_ylabel("Net Employment Growth (%)", fontweight="bold", labelpad=10)
        ax.set_ylim(-22, 45)
        ax.axhline(0, color="#64748B", linestyle="--", linewidth=1.1)

        self.render_plot(fig)

    # ==================== VIEW 5: WAGE PROGRESSION ====================
    def view_wage_trajectories(self):
        self.clear_panel()
        self.build_header_bar("Longitudinal Monthly Wage Trajectories by AI Exposure Tier")

        wage_pivot = self.df.pivot_table(index="Year", columns="Exposure_Tier", values="Wage_Num", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10.5, 5))
        fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.14)

        tier_colors = {"High": "#EF4444", "Moderate": "#D97706", "Low": "#10B981"}
        y_offsets = {"Moderate": 12, "High": -16, "Low": 12}

        for tier in ["High", "Moderate", "Low"]:
            if tier in wage_pivot.columns:
                ax.plot(wage_pivot.index, wage_pivot[tier], marker="o", markersize=7, linewidth=2.5, 
                        label=f"{tier} Exposure Tier", color=tier_colors[tier])
                for x, y in zip(wage_pivot.index, wage_pivot[tier]):
                    ax.annotate(f"₹{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, y_offsets[tier]),
                                ha="center", fontsize=8.5, fontweight="bold", color=tier_colors[tier])

        ax.set_ylim(26000, 56000)
        ax.set_title("Average Monthly Wage Progression by Exposure Tier (INR, 2019–2023)", fontweight="bold", pad=12)
        ax.set_xlabel("Year", fontweight="bold", labelpad=8)
        ax.set_ylabel("Average Monthly Wage (₹ INR)", fontweight="bold", labelpad=12)
        ax.set_xticks([2019, 2021, 2023])
        ax.legend(loc="upper left", frameon=True)

        self.render_plot(fig)

    # ==================== VIEW 6: EMERGING SKILLS ====================
    def view_emerging_skills(self):
        self.clear_panel()
        self.build_header_bar("Priority Emerging Technical Skills Ranked by Importance Score")

        skills = (
            self.df.groupby("Primary_Emerging_Skill")["Emerging_Skill_Importance_Score"]
            .mean()
            .reset_index()
            .sort_values(by="Emerging_Skill_Importance_Score", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(11, 5.2))
        fig.subplots_adjust(left=0.34, right=0.96, top=0.92, bottom=0.12)

        bars = ax.barh(skills["Primary_Emerging_Skill"], skills["Emerging_Skill_Importance_Score"], color="#059669", height=0.65)
        ax.set_title("Top 10 Emerging Technical Skills Across Indian Industries", fontweight="bold", pad=10)
        ax.set_xlabel("Mean Skill Importance Score (0 to 100)", fontweight="bold")
        ax.set_xlim(50, 100)
        ax.invert_yaxis()

        for bar, val in zip(bars, skills["Emerging_Skill_Importance_Score"]):
            ax.text(val + 0.8, bar.get_y() + bar.get_height()/2, f"{val:.1f}", 
                    ha="left", va="center", fontweight="bold", fontsize=8.5, color="#0F172A")

        self.render_plot(fig)

    # ==================== VIEW 7: INDUSTRY SENSITIVITY ====================
    def view_industry_analysis(self):
        self.clear_panel()
        self.build_header_bar("Top 10 Indian Industry Sectors Ranked by Mean AI Exposure")

        ind = (
            self.df.groupby("Industry_Sector")["AI_Exposure_Score"]
            .mean()
            .reset_index()
            .sort_values(by="AI_Exposure_Score", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(11, 5.2))
        fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.28)

        bars = ax.bar(ind["Industry_Sector"], ind["AI_Exposure_Score"], color="#059669", width=0.6)
        ax.set_title("Industry Exposure Index (Mean Score 0.0 to 1.0)", fontweight="bold", pad=10)
        ax.set_ylabel("Mean AI Exposure Score", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35, labelsize=9)

        for bar, val in zip(bars, ind["AI_Exposure_Score"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", 
                    ha="center", va="bottom", fontweight="bold", fontsize=8)

        self.render_plot(fig)

    # ==================== VIEW 8: CORRELATION MATRIX (CLEAR SLANTED LABELS) ====================
    def view_correlation_matrix(self):
        self.clear_panel()
        self.build_header_bar("Empirical Pearson Correlation Matrix (Proving Task Elasticity)")

        sub = self.occ_summary[["AI_Exposure_Score", "Automation_Potential_Pct", 
                                "Augmentation_Potential_Pct", "Net_Growth_Pct", "Wage_Growth_Pct"]].copy()
        sub.columns = ["AI Exposure", "Automation %", "Augmentation %", "Employment Growth %", "Wage Growth %"]
        corr = sub.corr()

        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        fig.subplots_adjust(left=0.22, right=0.96, top=0.90, bottom=0.22)

        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=0.8, ax=ax)
        ax.set_title("Correlation Matrix (r): Proving Negative Elasticity of Routine Tasks", fontweight="bold", pad=12)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

        self.render_plot(fig)

    # ==================== VIEW 9: PROPERLY ARRANGED ML SIMULATOR ====================
    def view_ml_simulator(self):
        self.clear_panel()
        self.build_header_bar("ML Predictive Simulator: Multi-Year Timeline Forecast to 2027")

        sim_container = tk.Frame(self.canvas_panel, bg="#FFFFFF")
        sim_container.pack(fill="both", expand=True, padx=16, pady=8)

        left_box = tk.Frame(sim_container, bg="#F8FAFC", bd=1, relief="solid", width=380, padx=16, pady=10)
        left_box.pack(side="left", fill="y", padx=(0, 12))
        left_box.pack_propagate(False)

        tk.Label(left_box, text="SELECT OCCUPATION:", font=("Segoe UI", 9, "bold"), fg="#1E293B", bg="#F8FAFC").pack(anchor="w")
        all_occupations = sorted(self.occ_summary["Occupation_Title"].unique().tolist())
        
        self.selected_occ_var = tk.StringVar(value="Computer Software Engineers" if "Computer Software Engineers" in all_occupations else all_occupations[0])
        occ_dropdown = ttk.Combobox(left_box, textvariable=self.selected_occ_var, values=all_occupations, state="readonly", width=32)
        occ_dropdown.pack(fill="x", pady=(2, 8))
        occ_dropdown.bind("<<ComboboxSelected>>", lambda e: self.load_selected_occupation_data())

        tk.Label(left_box, text="SCENARIO PARAMETERS (TWEAKABLE):", font=("Segoe UI", 9, "bold"), fg="#475569", bg="#F8FAFC").pack(anchor="w", pady=(2, 4))

        # Sliders
        tk.Label(left_box, text="AI Exposure Score (0.0 to 1.0):", font=("Segoe UI", 8, "bold"), bg="#F8FAFC").pack(anchor="w")
        self.sim_exposure_var = tk.DoubleVar()
        s1 = tk.Scale(left_box, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", 
                      variable=self.sim_exposure_var, bg="#F8FAFC", command=lambda e: self.update_ml_prediction_and_chart())
        s1.pack(fill="x", pady=(0, 4))

        tk.Label(left_box, text="Routine Automation Potential (%):", font=("Segoe UI", 8, "bold"), bg="#F8FAFC").pack(anchor="w")
        self.sim_auto_var = tk.DoubleVar()
        s2 = tk.Scale(left_box, from_=0.0, to=100.0, resolution=1.0, orient="horizontal", 
                      variable=self.sim_auto_var, bg="#F8FAFC", command=lambda e: self.sync_aug_slider(e))
        s2.pack(fill="x", pady=(0, 4))

        tk.Label(left_box, text="Cognitive Augmentation Potential (%):", font=("Segoe UI", 8, "bold"), bg="#F8FAFC").pack(anchor="w")
        self.sim_aug_var = tk.DoubleVar()
        s3 = tk.Scale(left_box, from_=0.0, to=100.0, resolution=1.0, orient="horizontal", 
                      variable=self.sim_aug_var, bg="#F8FAFC", command=lambda e: self.update_ml_prediction_and_chart())
        s3.pack(fill="x", pady=(0, 8))

        # Outcome Box
        tk.Label(left_box, text="PROJECTED 2027 HEADCOUNT & GROWTH", 
                 font=("Segoe UI", 8, "bold"), fg="#059669", bg="#F8FAFC").pack(anchor="w")
        self.pred_val_label = tk.Label(left_box, text="+0.00%", font=("Segoe UI", 24, "bold"), fg="#059669", bg="#F8FAFC")
        self.pred_val_label.pack(anchor="w", pady=(1, 2))

        self.pred_headcount_label = tk.Label(left_box, text="Forecast 2027: 0.0k workers", font=("Segoe UI", 10, "bold"), fg="#0F172A", bg="#F8FAFC")
        self.pred_headcount_label.pack(anchor="w", pady=(0, 3))

        self.verdict_title = tk.Label(left_box, text="Status: Running", font=("Segoe UI", 9, "bold"), fg="#0F172A", bg="#F8FAFC")
        self.verdict_title.pack(anchor="w", pady=(0, 3))

        self.verdict_desc = tk.Label(left_box, text="", font=("Segoe UI", 8), justify="left", fg="#475569", bg="#F8FAFC", wraplength=340)
        self.verdict_desc.pack(anchor="w", pady=(0, 4))

        # Right Live Forecast Chart Canvas
        self.chart_area = tk.Frame(sim_container, bg="#FFFFFF", bd=1, relief="solid")
        self.chart_area.pack(side="right", fill="both", expand=True)

        self.sim_canvas = None
        self.load_selected_occupation_data()

    def load_selected_occupation_data(self):
        selected_role = self.selected_occ_var.get()
        matched = self.occ_summary[self.occ_summary["Occupation_Title"] == selected_role]

        if not matched.empty:
            row = matched.iloc[0]
            self.sim_exposure_var.set(float(row["AI_Exposure_Score"]))
            self.sim_auto_var.set(float(row["Automation_Potential_Pct"]))
            self.sim_aug_var.set(float(row["Augmentation_Potential_Pct"]))
        self.update_ml_prediction_and_chart()

    def sync_aug_slider(self, val):
        auto_val = float(val)
        self.sim_aug_var.set(round(100.0 - auto_val, 1))
        self.update_ml_prediction_and_chart()

    def update_ml_prediction_and_chart(self):
        exp = self.sim_exposure_var.get()
        auto = self.sim_auto_var.get()
        aug = self.sim_aug_var.get()
        selected_role = self.selected_occ_var.get()

        try:
            w_2019 = float(self.occ_ts_data.loc[selected_role, 2019])
            w_2021 = float(self.occ_ts_data.loc[selected_role, 2021])
            w_2023 = float(self.occ_ts_data.loc[selected_role, 2023])
        except Exception:
            w_2019, w_2021, w_2023 = 80.0, 90.0, 100.0

        X_input = pd.DataFrame([[exp, auto, aug]], columns=["AI_Exposure_Score", "Automation_Potential_Pct", "Augmentation_Potential_Pct"])
        predicted_growth = self.ml_model.predict(X_input)[0]

        proj_workers_2027 = max(0.0, w_2023 * (1.0 + (predicted_growth / 100.0)))
        proj_workers_2025 = max(0.0, w_2023 * (1.0 + (predicted_growth / 200.0)))

        if predicted_growth >= 0:
            self.pred_val_label.config(text=f"+{predicted_growth:.1f}%", fg="#059669")
            self.pred_headcount_label.config(text=f"2027 Forecast: {proj_workers_2027:.1f}k workers", fg="#059669")
            self.verdict_title.config(text="Net Expansion Trajectory (2023–2027)", fg="#059669")
            self.verdict_desc.config(
                text=f"Augmentation elasticity for '{selected_role}' drives healthy expansion (+{proj_workers_2027 - w_2023:.1f}k workers by 2027)."
            )
        else:
            self.pred_val_label.config(text=f"{predicted_growth:.1f}%", fg="#DC2626")
            self.pred_headcount_label.config(text=f"2027 Forecast: {proj_workers_2027:.1f}k workers", fg="#DC2626")
            self.verdict_title.config(text="Contraction Warning (2023–2027)", fg="#DC2626")
            self.verdict_desc.config(
                text=f"Automation pressure for '{selected_role}' indicates displacement risk ({proj_workers_2027 - w_2023:.1f}k workers by 2027)."
            )

        if self.sim_canvas is not None:
            self.sim_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        fig.subplots_adjust(left=0.15, right=0.94, top=0.88, bottom=0.16)

        hist_years = [2019, 2021, 2023]
        hist_values = [w_2019, w_2021, w_2023]

        proj_years = [2023, 2025, 2027]
        proj_values = [w_2023, proj_workers_2025, proj_workers_2027]

        all_vals = hist_values + proj_values
        y_min = min(all_vals) * 0.88
        y_max = max(all_vals) * 1.20
        ax.set_ylim(y_min, y_max)

        # 1. Historical Actual Trendline
        ax.plot(hist_years, hist_values, color="#0F172A", marker="o", markersize=7, linewidth=2.5, label="Historical Panel (2019–2023)")
        ax.annotate(f"{w_2019:.1f}k", (2019, w_2019), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, fontweight="bold", color="#1E293B")
        ax.annotate(f"{w_2021:.1f}k", (2021, w_2021), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, fontweight="bold", color="#1E293B")

        # 2023 Baseline anchor
        ax.annotate(f"2023 Baseline\n{w_2023:.1f}k", (2023, w_2023), 
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5, fontweight="bold", color="#0F172A",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#F1F5F9", ec="#94A3B8"))

        # 2. Future Forecast Trendline
        future_color = "#059669" if predicted_growth >= 0 else "#DC2626"
        ax.plot(proj_years, proj_values, color=future_color, marker="s", markersize=8, linewidth=2.8, linestyle="--", label="Projected Horizon (2023–2027)")

        # 2025 Projected
        ax.annotate(f"{proj_workers_2025:.1f}k", (2025, proj_workers_2025),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, fontweight="bold", color=future_color)

        # 2027 Future Forecast
        ax.annotate(f"2027 FORECAST: {proj_workers_2027:.1f}k\n(4-Yr Shift: {predicted_growth:+.1f}%)", 
                    (2027, proj_workers_2027),
                    textcoords="offset points", xytext=(-10, 16), ha="right", fontsize=9, fontweight="bold", color=future_color,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec=future_color, lw=1.6))

        clean_role_title = (selected_role[:32] + "..") if len(selected_role) > 34 else selected_role
        ax.set_title(f"Forecast Horizon: {clean_role_title} (2019–2027)\nModel R² = {self.r2_metric:.2%}", 
                     fontweight="bold", fontsize=10.5, pad=12)

        ax.set_xticks([2019, 2021, 2023, 2025, 2027])
        ax.set_xticklabels(["2019", "2021", "2023\n(Baseline)", "2025\n(Projected)", "2027\n(Forecast)"], 
                           fontsize=8.5, fontweight="bold")
        
        ax.set_xlabel("Timeline", fontweight="bold", labelpad=6)
        ax.set_ylabel("Workforce Headcount (Thousands)", fontweight="bold", labelpad=12)
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)

        self.sim_canvas = FigureCanvasTkAgg(fig, master=self.chart_area)
        self.sim_canvas.draw()
        self.sim_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        plt.close(fig)

if __name__ == "__main__":
    app = AIWorkplaceExecutiveSuite()
    app.mainloop()