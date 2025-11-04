"""
Curva.io - AI-Powered Data Visualization Application
=====================================================
A comprehensive data visualization platform with intelligent AI recommendations.

Features:
- Multi-format file support (CSV, Excel)
- AI-powered visualization recommendations
- 8+ chart types
- Interactive visualizations
- Export capabilities
- Professional design
- Mobile responsive

Author: Curva.io Team
Version: 2.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import io


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Application configuration."""
    PAGE_TITLE: str = "Curva.io - Visualización Inteligente"
    PAGE_ICON: str = "📊"
    LAYOUT: str = "wide"
    MISTRAL_MODEL: str = "mistral-tiny"
    MISTRAL_API_URL: str = "https://api.mistral.ai/v1/chat/completions"
    MAX_UPLOAD_SIZE: int = 200  # MB
    
    # Color scheme based on Curva.io branding
    COLORS: Dict[str, str] = None
    
    def __post_init__(self):
        self.COLORS = {
            'navy_dark': '#1e3a5f',
            'navy_medium': '#2c5282',
            'blue_medium': '#4a90e2',
            'blue_light': '#5bc0de',
            'aqua': '#4ecdc4',
            'mint': '#72dfd0',
            'bg_light': '#f0f8ff',
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'accent': '#2ECC71',
            'danger': '#E74C3C',
        }


config = Config()


# ============================================================================
# STYLING
# ============================================================================

def load_custom_css() -> str:
    """Return comprehensive custom CSS for the application."""
    return """
    <style>
        /* ==================== RESET & BASE ==================== */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --navy-dark: #1e3a5f;
            --navy-medium: #2c5282;
            --blue-medium: #4a90e2;
            --blue-light: #5bc0de;
            --aqua: #4ecdc4;
            --mint: #72dfd0;
            --bg-light: #f0f8ff;
            --primary: #2C3E50;
            --secondary: #3498DB;
            --accent: #2ECC71;
            --danger: #E74C3C;
            --border-radius: 12px;
            --box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* ==================== MAIN APP STYLING ==================== */
        .stApp {
            background: linear-gradient(135deg, var(--bg-light) 0%, #ffffff 100%);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* ==================== HEADER ==================== */
        .main-header {
            background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-medium) 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
            animation: fadeInDown 0.8s ease;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            animation: fadeIn 1s ease;
        }
        
        .logo-icon {
            width: 60px;
            height: 60px;
            margin-right: 15px;
            animation: rotate 20s linear infinite;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .logo-text {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(90deg, var(--blue-light) 0%, var(--aqua) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            letter-spacing: 2px;
        }
        
        .subtitle {
            color: white;
            text-align: center;
            font-size: 1.3rem;
            opacity: 0.95;
            font-weight: 300;
            letter-spacing: 1px;
        }
        
        .hero-badges {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        
        .badge {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 8px 20px;
            border-radius: 50px;
            color: white;
            font-size: 0.9rem;
            font-weight: 500;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: var(--transition);
        }
        
        .badge:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        /* ==================== SECTION CONTAINERS ==================== */
        .section-container {
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            border-left: 5px solid var(--blue-medium);
            animation: fadeInUp 0.6s ease;
            transition: var(--transition);
        }
        
        .section-container:hover {
            box-shadow: 0 12px 35px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        
        .section-title {
            color: var(--navy-dark);
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* ==================== BUTTONS ==================== */
        .stButton > button {
            background: linear-gradient(135deg, var(--blue-medium) 0%, var(--blue-light) 100%);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 0.9rem 2.5rem;
            font-weight: 600;
            font-size: 1.05rem;
            transition: var(--transition);
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(74, 144, 226, 0.5);
            background: linear-gradient(135deg, var(--blue-light) 0%, var(--aqua) 100%);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }
        
        /* ==================== FILE UPLOADER ==================== */
        .stFileUploader {
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.05) 0%, rgba(78, 205, 196, 0.05) 100%);
            border: 3px dashed var(--blue-medium);
            border-radius: 20px;
            padding: 2.5rem;
            transition: var(--transition);
        }
        
        .stFileUploader:hover {
            border-color: var(--aqua);
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
            transform: translateY(-2px);
        }
        
        [data-testid="stFileUploadDropzone"] {
            background: transparent;
        }
        
        /* ==================== METRICS CARDS ==================== */
        .metric-card {
            background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-medium) 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, transparent 0%, rgba(255,255,255,0.1) 100%);
            pointer-events: none;
        }
        
        .metric-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .metric-value {
            font-size: 2.8rem;
            font-weight: 800;
            color: var(--aqua);
            text-shadow: 0 2px 10px rgba(78, 205, 196, 0.3);
            position: relative;
            z-index: 1;
        }
        
        .metric-label {
            font-size: 1.1rem;
            opacity: 0.95;
            margin-top: 0.5rem;
            font-weight: 500;
            position: relative;
            z-index: 1;
        }
        
        /* ==================== DATAFRAME ==================== */
        .dataframe {
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* ==================== SELECTBOX & INPUTS ==================== */
        .stSelectbox > div > div,
        .stTextInput > div > div,
        .stTextArea > div > div {
            background: var(--bg-light);
            border-radius: 12px;
            border: 2px solid transparent;
            transition: var(--transition);
        }
        
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div:focus-within,
        .stTextArea > div > div:focus-within {
            border-color: var(--blue-medium);
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
            background: white;
        }
        
        /* ==================== TABS ==================== */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-light);
            border-radius: 15px;
            padding: 0.8rem;
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            color: var(--navy-dark);
            font-weight: 600;
            padding: 12px 24px;
            transition: var(--transition);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(74, 144, 226, 0.1);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--blue-medium) 0%, var(--blue-light) 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        }
        
        /* ==================== EXPANDER ==================== */
        .streamlit-expanderHeader {
            background: var(--bg-light);
            border-radius: 12px;
            font-weight: 600;
            color: var(--navy-dark);
            transition: var(--transition);
        }
        
        .streamlit-expanderHeader:hover {
            background: var(--blue-light);
            color: white;
        }
        
        /* ==================== INFO/SUCCESS/WARNING BOXES ==================== */
        .stAlert {
            border-radius: 15px;
            border-left: 5px solid;
            padding: 1.2rem;
            animation: slideIn 0.5s ease;
        }
        
        [data-testid="stNotificationContentInfo"] {
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
            border-left-color: var(--blue-medium);
        }
        
        [data-testid="stNotificationContentSuccess"] {
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
            border-left-color: var(--accent);
        }
        
        [data-testid="stNotificationContentWarning"] {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
            border-left-color: #FFC107;
        }
        
        [data-testid="stNotificationContentError"] {
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(192, 57, 43, 0.1) 100%);
            border-left-color: var(--danger);
        }
        
        /* ==================== SPINNER ==================== */
        .stSpinner > div {
            border-top-color: var(--blue-medium) !important;
        }
        
        /* ==================== ANIMATIONS ==================== */
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* ==================== WAVE ANIMATION ==================== */
        .wave-animation {
            position: relative;
            height: 120px;
            margin: -50px 0 20px 0;
            overflow: hidden;
        }
        
        .wave-animation::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 200%;
            height: 100%;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%234ecdc4' fill-opacity='0.3' d='M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,122.7C672,117,768,139,864,133.3C960,128,1056,96,1152,90.7C1248,85,1344,107,1392,117.3L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
            background-size: 50% 100%;
            animation: wave 15s linear infinite;
        }
        
        @keyframes wave {
            0% {
                transform: translateX(0);
            }
            100% {
                transform: translateX(-50%);
            }
        }
        
        /* ==================== RESPONSIVE DESIGN ==================== */
        @media (max-width: 768px) {
            .logo-text {
                font-size: 2.5rem;
            }
            
            .subtitle {
                font-size: 1rem;
            }
            
            .section-container {
                padding: 1.5rem;
            }
            
            .metric-value {
                font-size: 2rem;
            }
            
            .hero-badges {
                flex-direction: column;
                align-items: center;
            }
        }
        
        @media (max-width: 480px) {
            .logo-text {
                font-size: 2rem;
            }
            
            .main-header {
                padding: 1.5rem;
            }
            
            .section-container {
                padding: 1rem;
            }
        }
        
        /* ==================== CUSTOM SCROLLBAR ==================== */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-light);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--blue-medium), var(--aqua));
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, var(--navy-dark), var(--blue-medium));
        }
    </style>
    """


# ============================================================================
# AI ANALYZER
# ============================================================================

class MistralAnalyzer:
    """Handle AI-powered data analysis using Mistral AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral analyzer.
        
        Args:
            api_key: Optional Mistral API key
        """
        self.api_key = api_key
        self.base_url = config.MISTRAL_API_URL
        self.model = config.MISTRAL_MODEL
    
    def analyze_data(self, df_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data and suggest optimal visualization.
        
        Args:
            df_summary: Summary of the dataframe
            
        Returns:
            Dictionary with visualization recommendations
        """
        if not self.api_key:
            return self._get_basic_suggestion(df_summary)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = self._create_analysis_prompt(df_summary)
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return self._parse_ai_response(content)
            else:
                st.warning(f"API returned status code {response.status_code}")
                return self._get_basic_suggestion(df_summary)
                
        except requests.exceptions.Timeout:
            st.warning("AI analysis timed out. Using basic analysis.")
            return self._get_basic_suggestion(df_summary)
        except Exception as e:
            st.warning(f"AI analysis failed: {str(e)}. Using basic analysis.")
            return self._get_basic_suggestion(df_summary)
    
    def _create_analysis_prompt(self, df_summary: Dict[str, Any]) -> str:
        """Create analysis prompt for the AI."""
        return f"""
        Analiza los siguientes datos y sugiere el MEJOR tipo de visualización.
        
        Resumen de datos:
        - Filas: {df_summary['shape'][0]}
        - Columnas: {df_summary['shape'][1]}
        - Columnas numéricas: {df_summary['numeric_columns']}
        - Columnas categóricas: {df_summary['categorical_columns']}
        - Muestra: {json.dumps(df_summary['sample'], default=str)}
        
        Responde SOLO con un JSON válido en este formato:
        {{
            "visualization_type": "uno de: [line, bar, scatter, pie, heatmap, box, area, histogram]",
            "reason": "explicación breve en español (máximo 100 caracteres)",
            "x_column": "nombre de columna para eje X o null",
            "y_column": "nombre de columna para eje Y o null",
            "color_column": "columna para color o null",
            "insights": "insight clave sobre los datos (máximo 150 caracteres)"
        }}
        """
    
    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response and extract JSON."""
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback to basic suggestion
        return self._get_basic_suggestion({})
    
    def _get_basic_suggestion(self, df_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate basic visualization suggestion without AI.
        
        Args:
            df_summary: Summary of the dataframe
            
        Returns:
            Dictionary with basic visualization recommendations
        """
        numeric_cols = df_summary.get('numeric_columns', [])
        categorical_cols = df_summary.get('categorical_columns', [])
        
        # Determine best visualization type
        if len(numeric_cols) >= 2:
            viz_type = "scatter"
            reason = "Múltiples variables numéricas - ideal para dispersión"
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            viz_type = "bar"
            reason = "Datos categóricos y numéricos - ideal para comparaciones"
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
        elif len(numeric_cols) == 1:
            viz_type = "histogram"
            reason = "Una variable numérica - distribución con histograma"
            x_col = None
            y_col = numeric_cols[0]
        else:
            viz_type = "bar"
            reason = "Visualización estándar para sus datos"
            x_col = df_summary.get('columns', [None])[0]
            y_col = None
        
        return {
            "visualization_type": viz_type,
            "reason": reason,
            "x_column": x_col,
            "y_column": y_col,
            "color_column": None,
            "insights": "Análisis básico completado"
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class VisualizationEngine:
    """Handle all visualization creation and management."""
    
    # Color palette based on Curva.io branding
    COLORS = [
        '#1e3a5f', '#4a90e2', '#5bc0de', '#4ecdc4', '#72dfd0',
        '#2c5282', '#3498db', '#5dade2', '#85c1e2', '#aed6f1'
    ]
    
    CHART_TYPES = {
        "line": "📈 Gráfico de Líneas",
        "bar": "📊 Gráfico de Barras",
        "scatter": "⚡ Diagrama de Dispersión",
        "pie": "🥧 Gráfico Circular",
        "heatmap": "🔥 Mapa de Calor",
        "box": "📦 Diagrama de Caja",
        "area": "🏔️ Gráfico de Área",
        "histogram": "📊 Histograma"
    }
    
    @staticmethod
    def create_visualization(
        df: pd.DataFrame,
        viz_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        title: str = ""
    ) -> Optional[go.Figure]:
        """
        Create visualization based on specified type.
        
        Args:
            df: DataFrame to visualize
            viz_type: Type of visualization
            x_col: Column for X axis
            y_col: Column for Y axis
            color_col: Column for color encoding
            title: Chart title
            
        Returns:
            Plotly figure object or None if creation fails
        """
        try:
            # Auto-select columns if not provided
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if x_col is None and len(df.columns) > 0:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
            if y_col is None and len(numeric_cols) > 0:
                y_col = numeric_cols[0]
            
            # Limit data for performance
            df_plot = df.head(1000) if len(df) > 1000 else df
            
            # Create visualization based on type
            fig = None
            
            if viz_type == "line":
                fig = px.line(
                    df_plot, x=x_col, y=y_col, color=color_col,
                    title=title or "Gráfico de Líneas",
                    color_discrete_sequence=VisualizationEngine.COLORS
                )
            
            elif viz_type == "bar":
                fig = px.bar(
                    df_plot, x=x_col, y=y_col, color=color_col,
                    title=title or "Gráfico de Barras",
                    color_discrete_sequence=VisualizationEngine.COLORS
                )
            
            elif viz_type == "scatter":
                fig = px.scatter(
                    df_plot, x=x_col, y=y_col, color=color_col,
                    title=title or "Diagrama de Dispersión",
                    color_discrete_sequence=VisualizationEngine.COLORS,
                    opacity=0.7
                )
            
            elif viz_type == "pie":
                if y_col and x_col:
                    # Aggregate data for pie chart
                    pie_data = df_plot.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.pie(
                        pie_data, values=y_col, names=x_col,
                        title=title or "Gráfico Circular",
                        color_discrete_sequence=VisualizationEngine.COLORS
                    )
            
            elif viz_type == "heatmap":
                # Create correlation matrix for numeric columns
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale=[
                            [0, '#1e3a5f'],
                            [0.5, '#4a90e2'],
                            [1, '#4ecdc4']
                        ],
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    fig.update_layout(title=title or "Mapa de Calor - Correlación")
            
            elif viz_type == "box":
                fig = px.box(
                    df_plot, x=x_col, y=y_col, color=color_col,
                    title=title or "Diagrama de Caja",
                    color_discrete_sequence=VisualizationEngine.COLORS
                )
            
            elif viz_type == "area":
                fig = px.area(
                    df_plot, x=x_col, y=y_col, color=color_col,
                    title=title or "Gráfico de Área",
                    color_discrete_sequence=VisualizationEngine.COLORS
                )
            
            elif viz_type == "histogram":
                fig = px.histogram(
                    df_plot, x=y_col if y_col else x_col,
                    title=title or "Histograma",
                    color_discrete_sequence=VisualizationEngine.COLORS,
                    nbins=30
                )
            
            if fig:
                # Apply Curva.io styling
                fig.update_layout(
                    font=dict(family="Inter, Arial, sans-serif", size=12),
                    plot_bgcolor='rgba(240, 248, 255, 0.5)',
                    paper_bgcolor='white',
                    title_font_size=20,
                    title_font_color='#1e3a5f',
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="#4a90e2",
                        borderwidth=1,
                        font=dict(size=11)
                    ),
                    margin=dict(l=50, r=50, t=80, b=50),
                    height=500
                )
                
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(74, 144, 226, 0.1)',
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(74, 144, 226, 0.3)'
                )
                
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(74, 144, 226, 0.1)',
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(74, 144, 226, 0.3)'
                )
            
            return fig
        
        except Exception as e:
            st.error(f"Error al crear visualización: {str(e)}")
            return None
    
    @staticmethod
    def export_figure(fig: go.Figure, format: str = "png") -> bytes:
        """
        Export figure to specified format.
        
        Args:
            fig: Plotly figure to export
            format: Export format (png, jpg, pdf, html)
            
        Returns:
            Bytes of the exported figure
        """
        try:
            if format in ["png", "jpg", "jpeg"]:
                return fig.to_image(
                    format=format,
                    width=1200,
                    height=800,
                    scale=2
                )
            elif format == "pdf":
                return fig.to_image(format="pdf")
            elif format == "html":
                return fig.to_html().encode()
            else:
                return fig.to_image(format="png")
        except Exception as e:
            st.error(f"Error al exportar: {str(e)}")
            return b""


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Handle data loading and processing."""
    
    @staticmethod
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame or None if loading fails
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            
            else:
                st.error("Formato de archivo no soportado")
                return None
            
            # Basic data cleaning
            df = DataProcessor.clean_data(df)
            
            return df
        
        except Exception as e:
            st.error(f"Error al cargar archivo: {str(e)}")
            return None
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Convert object columns to category if they have few unique values
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(5).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # MB
        }


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render the application header."""
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-container">
            <svg class="logo-icon" width="60" height="60" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="url(#gradient)"/>
                <path d="M 20 50 Q 35 30 50 50 T 80 50" stroke="white" stroke-width="3" fill="none"/>
                <circle cx="30" cy="40" r="3" fill="white"/>
                <circle cx="50" cy="50" r="3" fill="white"/>
                <circle cx="70" cy="40" r="3" fill="white"/>
                <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="{config.COLORS['navy_dark']}"/>
                        <stop offset="100%" stop-color="{config.COLORS['blue_medium']}"/>
                    </linearGradient>
                </defs>
            </svg>
            <div class="logo-text">Curva.io</div>
        </div>
        <div class="subtitle">Visualización Inteligente de Datos con IA</div>
        <div class="hero-badges">
            <span class="badge">🔒 100% Privado</span>
            <span class="badge">⚡ Tiempo Real</span>
            <span class="badge">🎁 Gratuito</span>
            <span class="badge">🤖 IA Integrada</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(df: pd.DataFrame):
    """
    Render data metrics cards.
    
    Args:
        df: DataFrame to display metrics for
    """
    col1, col2, col3, col4 = st.columns(4)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    null_count = df.isnull().sum().sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Filas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Columnas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(numeric_cols)}</div>
            <div class="metric-label">Numéricas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{null_count:,}</div>
            <div class="metric-label">Valores Nulos</div>
        </div>
        """, unsafe_allow_html=True)


def render_download_button(fig: go.Figure, format: str, key: str):
    """
    Render download button for figure export.
    
    Args:
        fig: Figure to export
        format: Export format
        key: Unique key for button
    """
    try:
        img_bytes = VisualizationEngine.export_figure(fig, format)
        
        if img_bytes:
            mime_types = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'pdf': 'application/pdf',
                'html': 'text/html'
            }
            
            st.download_button(
                label=f"📥 {format.upper()}",
                data=img_bytes,
                file_name=f"curva_io_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}",
                mime=mime_types.get(format, 'application/octet-stream'),
                key=key,
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error al preparar descarga: {str(e)}")


def render_footer():
    """Render application footer."""
    st.markdown("""
    <div class="wave-animation"></div>
    <div style="text-align: center; padding: 2rem; color: #1e3a5f;">
        <p style="font-size: 0.95rem; opacity: 0.8; margin-bottom: 0.5rem;">
            <strong>Curva.io</strong> © 2024 | Visualización Inteligente de Datos
        </p>
        <p style="font-size: 0.85rem; opacity: 0.6;">
            Hecho con ❤️ para democratizar los datos
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'ai_suggestion' not in st.session_state:
        st.session_state.ai_suggestion = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    st.markdown(load_custom_css(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # ========================================================================
    # SECTION 1: DATA UPLOAD
    # ========================================================================
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Paso 1: Cargar Datos</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona tu archivo de datos",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel (XLSX, XLS)",
        label_visibility="collapsed"
    )
    
    # API Configuration (Optional)
    with st.expander("⚙️ Configuración de IA (Opcional)", expanded=False):
        st.markdown("""
        **Mistral AI** proporciona análisis inteligente de datos.  
        Si no tienes una clave API, la aplicación usará análisis básico automático.
        """)
        api_key = st.text_input(
            "Clave API de Mistral AI",
            type="password",
            help="Ingresa tu clave API de Mistral para análisis avanzado con IA",
            placeholder="sk-..."
        )
        st.markdown("[🔗 Obtener clave API de Mistral](https://console.mistral.ai/)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 2: DATA ANALYSIS
    # ========================================================================
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Cargando datos..."):
                df = DataProcessor.load_file(uploaded_file)
            
            if df is None:
                st.stop()
            
            st.session_state.data = df
            st.session_state.data_summary = DataProcessor.get_data_summary(df)
            
            # Display metrics
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Paso 2: Análisis de Datos</div>', unsafe_allow_html=True)
            
            render_metrics(df)
            
            st.markdown("") # Spacing
            
            # Data preview
            with st.expander("👁️ Vista previa de datos (primeras 10 filas)", expanded=True):
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    height=300
                )
            
            # AI Analysis section
            st.markdown("#### 🤖 Análisis Inteligente")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("""
                Nuestro sistema analizará tus datos y sugerirá la mejor visualización.
                """)
            
            with col2:
                analyze_btn = st.button(
                    "🚀 Analizar con IA",
                    key="analyze_btn",
                    use_container_width=True,
                    type="primary"
                )
            
            if analyze_btn:
                with st.spinner("🔍 Analizando datos con IA..."):
                    analyzer = MistralAnalyzer(api_key if api_key else None)
                    suggestion = analyzer.analyze_data(st.session_state.data_summary)
                    st.session_state.ai_suggestion = suggestion
                
                st.success("✅ Análisis completado")
                st.balloons()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ================================================================
            # SECTION 3: VISUALIZATION
            # ================================================================
            
            if st.session_state.ai_suggestion or st.button(
                "📈 Crear Visualización Básica",
                type="secondary"
            ):
                # Set default suggestion if not available
                if not st.session_state.ai_suggestion:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.session_state.ai_suggestion = {
                        "visualization_type": "bar",
                        "reason": "Visualización por defecto",
                        "x_column": df.columns[0],
                        "y_column": numeric_cols[0] if numeric_cols else None,
                        "color_column": None,
                        "insights": "Análisis básico"
                    }
                
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🎨 Paso 3: Visualización</div>', unsafe_allow_html=True)
                
                # Tabs for different visualizations
                tab1, tab2, tab3 = st.tabs([
                    "📊 Visualización Recomendada",
                    "🔄 Comparar con Otro Tipo",
                    "📋 Resumen de Datos"
                ])
                
                # TAB 1: Recommended Visualization
                with tab1:
                    suggestion = st.session_state.ai_suggestion
                    
                    # Display AI recommendation
                    st.info(f"""
                    **💡 Recomendación IA:**  
                    {suggestion.get('reason', 'Visualización recomendada')}
                    
                    **🎯 Insight:** {suggestion.get('insights', 'Datos listos para visualizar')}
                    """)
                    
                    # Create visualization
                    fig1 = VisualizationEngine.create_visualization(
                        df,
                        suggestion['visualization_type'],
                        suggestion.get('x_column'),
                        suggestion.get('y_column'),
                        suggestion.get('color_column'),
                        f"Visualización Recomendada: {VisualizationEngine.CHART_TYPES.get(suggestion['visualization_type'], '')}"
                    )
                    
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Download options
                        st.markdown("#### 💾 Descargar Visualización")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            render_download_button(fig1, 'png', 'dl_png_1')
                        with col2:
                            render_download_button(fig1, 'jpg', 'dl_jpg_1')
                        with col3:
                            render_download_button(fig1, 'pdf', 'dl_pdf_1')
                        with col4:
                            render_download_button(fig1, 'html', 'dl_html_1')
                
                # TAB 2: Comparison Visualization
                with tab2:
                    st.markdown("#### 🔄 Selecciona otro tipo de visualización")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        selected_viz = st.selectbox(
                            "Tipo de visualización:",
                            options=list(VisualizationEngine.CHART_TYPES.keys()),
                            format_func=lambda x: VisualizationEngine.CHART_TYPES[x],
                            key="comparison_select"
                        )
                    
                    # Column selection
                    st.markdown("##### Configuración de Ejes")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        x_col = st.selectbox(
                            "Eje X:",
                            options=[None] + list(df.columns),
                            key="x_select"
                        )
                    
                    with col2:
                        y_col = st.selectbox(
                            "Eje Y:",
                            options=[None] + list(df.columns),
                            key="y_select"
                        )
                    
                    with col3:
                        color_col = st.selectbox(
                            "Color:",
                            options=[None] + list(df.columns),
                            key="color_select"
                        )
                    
                    if st.button("🎨 Crear Comparación", type="primary", use_container_width=True):
                        fig2 = VisualizationEngine.create_visualization(
                            df,
                            selected_viz,
                            x_col,
                            y_col,
                            color_col,
                            f"Visualización Alternativa: {VisualizationEngine.CHART_TYPES[selected_viz]}"
                        )
                        
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Download options
                            st.markdown("#### 💾 Descargar Visualización")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                render_download_button(fig2, 'png', 'dl_png_2')
                            with col2:
                                render_download_button(fig2, 'jpg', 'dl_jpg_2')
                            with col3:
                                render_download_button(fig2, 'pdf', 'dl_pdf_2')
                            with col4:
                                render_download_button(fig2, 'html', 'dl_html_2')
                
                # TAB 3: Data Summary
                with tab3:
                    st.markdown("#### 📊 Estadísticas Descriptivas")
                    
                    # Numeric columns statistics
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.markdown("##### Variables Numéricas")
                        st.dataframe(
                            df[numeric_cols].describe(),
                            use_container_width=True
                        )
                    
                    # Categorical columns info
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        st.markdown("##### Variables Categóricas")
                        for col in categorical_cols[:5]:  # Show first 5
                            with st.expander(f"📌 {col}"):
                                value_counts = df[col].value_counts().head(10)
                                st.bar_chart(value_counts)
                    
                    # Missing values
                    st.markdown("##### Valores Faltantes")
                    missing = df.isnull().sum()
                    missing = missing[missing > 0]
                    if len(missing) > 0:
                        st.bar_chart(missing)
                    else:
                        st.success("✅ No hay valores faltantes")
                    
                    # Data types
                    st.markdown("##### Tipos de Datos")
                    dtype_df = pd.DataFrame({
                        'Columna': df.dtypes.index,
                        'Tipo': df.dtypes.values.astype(str)
                    })
                    st.dataframe(dtype_df, use_container_width=True, hide_index=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.info("Por favor, verifica que tu archivo esté en el formato correcto (CSV o Excel)")
            import traceback
            with st.expander("🔍 Detalles del error (para debugging)"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome screen when no file is uploaded
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 **¿Cómo empezar?**
        
        1. **📁 Carga tu archivo** de datos (CSV o Excel) usando el área de arriba
        2. **🤖 Analiza automáticamente** tus datos con IA
        3. **📊 Visualiza** con la recomendación inteligente
        4. **🔄 Compara** con otros tipos de gráficos
        5. **💾 Descarga** en el formato que prefieras
        
        ---
        
        ### ✨ **Características principales:**
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(74, 144, 226, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%); border-radius: 12px; border-left: 4px solid #4a90e2;">
                <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">🤖 IA Inteligente</h4>
                <p style="color: #6C757D; margin: 0;">Análisis automático y recomendaciones de visualización</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(46, 204, 113, 0.1) 100%); border-radius: 12px; border-left: 4px solid #4ecdc4;">
                <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">📊 8+ Visualizaciones</h4>
                <p style="color: #6C757D; margin: 0;">Líneas, barras, dispersión, mapas de calor y más</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(52, 152, 219, 0.1) 100%); border-radius: 12px; border-left: 4px solid #2ECC71;">
                <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">🎨 Diseño Profesional</h4>
                <p style="color: #6C757D; margin: 0;">Gráficos de alta calidad listos para presentar</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(74, 144, 226, 0.1) 100%); border-radius: 12px; border-left: 4px solid #3498DB;">
                <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">📱 100% Responsive</h4>
                <p style="color: #6C757D; margin: 0;">Funciona perfectamente en cualquier dispositivo</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(74, 144, 226, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%); border-radius: 12px; border-left: 4px solid #4a90e2;">
                <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">💾 Múltiples Formatos</h4>
                <p style="color: #6C757D; margin: 0;">Descarga en PNG, JPG, PDF o HTML interactivo</p>
            </div>
            <div style="padding: 1.5rem; background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(46, 204, 113, 0.1) 100%); border-radius: 12px; border-left: 4px solid #4ecdc4;">
                <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">🔒 Privacidad Total</h4>
                <p style="color: #6C757D; margin: 0;">Tus datos nunca se guardan en nuestros servidores</p>
            </div>
        </div>
        
        ---
        
        ### 📚 **Formatos Soportados:**
        - **CSV** (.csv) - Archivos de valores separados por comas
        - **Excel** (.xlsx, .xls) - Hojas de cálculo de Microsoft Excel
        
        ### 🎯 **Tipos de Visualización Disponibles:**
        - 📈 Gráficos de Líneas - Para tendencias temporales
        - 📊 Gráficos de Barras - Para comparaciones
        - ⚡ Diagramas de Dispersión - Para correlaciones
        - 🥧 Gráficos Circulares - Para distribuciones
        - 🔥 Mapas de Calor - Para correlaciones múltiples
        - 📦 Diagramas de Caja - Para distribuciones estadísticas
        - 🏔️ Gráficos de Área - Para acumulaciones
        - 📊 Histogramas - Para frecuencias
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Render footer
    render_footer()


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
