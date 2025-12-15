"""
Curva.io - Data Visualization Platform with AI Assistance
A Flask application for intelligent data visualization
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import io
import base64
from anthropic import Anthropic
import plotly.io as pio
from datetime import datetime
import tempfile
from typing import Optional, Dict, Any, List, Tuple

# =============================================================================
# Application Configuration
# =============================================================================

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.xls']
app.config['MAX_ROWS_FOR_ANALYSIS'] = 10000  # Limit rows for AI analysis
app.config['MAX_EXCEL_SHEETS'] = 1  # Only allow single-sheet Excel files

# API Configuration
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', 'your-api-key-here')
anthropic = Anthropic(api_key=CLAUDE_API_KEY)

# Visualization Types (Spanish labels)
VISUALIZATION_TYPES = {
    'bar': 'Gráfico de Barras',
    'line': 'Gráfico de Líneas',
    'scatter': 'Diagrama de Dispersión',
    'pie': 'Gráfico Circular',
    'heatmap': 'Mapa de Calor',
    'box': 'Diagrama de Caja',
    'histogram': 'Histograma',
    'area': 'Gráfico de Área',
    'bubble': 'Gráfico de Burbujas',
    'treemap': 'Mapa de Árbol',
    'sunburst': 'Gráfico Sunburst',
    'violin': 'Gráfico de Violín',
    'radar': 'Gráfico Radar',
    'waterfall': 'Gráfico en Cascada',
    'funnel': 'Gráfico de Embudo'
}

# =============================================================================
# Data Processing Utilities
# =============================================================================

def validate_excel_file(file) -> Tuple[bool, str, Optional[List[str]]]:
    """
    Validate Excel file and check for multiple sheets.
    
    Returns:
        Tuple of (is_valid, error_message, sheet_names)
    """
    try:
        # Read Excel file to get sheet names without loading all data
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        
        if len(sheet_names) == 0:
            return False, "El archivo Excel está vacío o corrupto.", None
        
        if len(sheet_names) > 1:
            return False, (
                f"El archivo Excel contiene {len(sheet_names)} hojas: "
                f"{', '.join(sheet_names[:5])}{'...' if len(sheet_names) > 5 else ''}. "
                f"Por favor, sube un archivo con una sola hoja de datos, "
                f"o exporta la hoja deseada como CSV."
            ), sheet_names
        
        return True, "", sheet_names
        
    except Exception as e:
        return False, f"Error al leer el archivo Excel: {str(e)}", None


def read_data_file(file, file_ext: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Read data file (CSV or Excel) with proper error handling.
    
    Returns:
        Tuple of (dataframe, error_message)
    """
    try:
        if file_ext == '.csv':
            # Try multiple encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                return None, "No se pudo decodificar el archivo CSV. Intenta guardarlo con codificación UTF-8."
                
        else:  # Excel files
            file.seek(0)
            
            # Validate Excel file first
            is_valid, error_msg, sheet_names = validate_excel_file(file)
            
            if not is_valid:
                return None, error_msg
            
            # Read the single sheet
            file.seek(0)
            df = pd.read_excel(file, sheet_name=0)
        
        # Validate dataframe
        if df is None or df.empty:
            return None, "El archivo está vacío o no contiene datos válidos."
        
        if len(df.columns) == 0:
            return None, "No se encontraron columnas en el archivo."
        
        if len(df) == 0:
            return None, "El archivo no contiene filas de datos."
        
        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return None, "Después de limpiar datos vacíos, no quedan datos válidos."
        
        return df, ""
        
    except pd.errors.EmptyDataError:
        return None, "El archivo está vacío."
    except pd.errors.ParserError as e:
        return None, f"Error al analizar el archivo: {str(e)}"
    except Exception as e:
        return None, f"Error inesperado al leer el archivo: {str(e)}"


def prepare_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare comprehensive data summary for AI analysis.
    """
    # Detect column types more accurately
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to detect datetime columns stored as strings
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col], errors='raise')
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except (ValueError, TypeError):
            pass
    
    # Calculate statistics for numeric columns
    numeric_stats = {}
    for col in numeric_cols[:5]:  # Limit to first 5
        numeric_stats[col] = {
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'unique_count': int(df[col].nunique())
        }
    
    # Get categorical column info
    categorical_info = {}
    for col in categorical_cols[:5]:  # Limit to first 5
        unique_vals = df[col].dropna().unique()[:10]  # First 10 unique values
        categorical_info[col] = {
            'unique_count': int(df[col].nunique()),
            'sample_values': [str(v) for v in unique_vals]
        }
    
    # Detect potential relationships
    has_time_series = len(datetime_cols) > 0 or any(
        'fecha' in col.lower() or 'date' in col.lower() or 
        'time' in col.lower() or 'año' in col.lower() or 
        'mes' in col.lower() or 'year' in col.lower()
        for col in df.columns
    )
    
    has_categories = len(categorical_cols) > 0 and any(
        df[col].nunique() <= 20 for col in categorical_cols
    )
    
    return {
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'datetime_columns': datetime_cols,
        'numeric_stats': numeric_stats,
        'categorical_info': categorical_info,
        'has_time_series': has_time_series,
        'has_categories': has_categories,
        'missing_values': {col: int(df[col].isna().sum()) for col in df.columns},
        'sample_data': df.head(3).to_dict('records')
    }


# =============================================================================
# AI Analysis
# =============================================================================

def analyze_data_with_ai(df: pd.DataFrame, context: str) -> Dict[str, Any]:
    """
    Use Claude to analyze data and suggest the best visualization.
    """
    try:
        data_summary = prepare_data_summary(df)
        
        prompt = f"""Eres un experto en visualización de datos. Analiza este dataset y sugiere la mejor visualización.

INFORMACIÓN DEL DATASET:
- Dimensiones: {data_summary['shape']['rows']} filas × {data_summary['shape']['columns']} columnas
- Columnas: {', '.join(data_summary['columns'])}

TIPOS DE COLUMNAS:
- Numéricas: {', '.join(data_summary['numeric_columns']) or 'Ninguna'}
- Categóricas: {', '.join(data_summary['categorical_columns']) or 'Ninguna'}
- Fechas/Tiempo: {', '.join(data_summary['datetime_columns']) or 'Ninguna'}

ESTADÍSTICAS NUMÉRICAS:
{json.dumps(data_summary['numeric_stats'], indent=2, ensure_ascii=False)}

INFORMACIÓN CATEGÓRICA:
{json.dumps(data_summary['categorical_info'], indent=2, ensure_ascii=False)}

CARACTERÍSTICAS DETECTADAS:
- ¿Datos de series temporales?: {'Sí' if data_summary['has_time_series'] else 'No'}
- ¿Tiene categorías para agrupar?: {'Sí' if data_summary['has_categories'] else 'No'}

MUESTRA DE DATOS (primeras 3 filas):
{json.dumps(data_summary['sample_data'], indent=2, ensure_ascii=False)}

CONTEXTO DEL USUARIO: {context if context else 'No proporcionado'}

REGLAS DE SELECCIÓN:
1. Para series temporales → line o area
2. Para comparar categorías → bar (horizontal si hay muchas categorías)
3. Para distribuciones → histogram o box
4. Para correlaciones entre 2 variables numéricas → scatter
5. Para proporciones de un total → pie (máximo 6-8 categorías)
6. Para múltiples variables numéricas correlacionadas → heatmap
7. Para mostrar rangos y outliers → box o violin
8. Para jerarquías → treemap o sunburst
9. Para procesos con etapas → funnel
10. Para cambios incrementales → waterfall

Responde ÚNICAMENTE con un objeto JSON válido (sin markdown, sin explicaciones adicionales):
{{
    "visualization_type": "tipo elegido de: [bar, line, scatter, pie, heatmap, box, histogram, area, bubble, treemap, sunburst, violin, radar, waterfall, funnel]",
    "reasoning": "explicación breve en español de por qué este tipo es el mejor",
    "x_column": "columna sugerida para eje X o null",
    "y_column": "columna sugerida para eje Y o null",
    "color_column": "columna para agrupar por color o null",
    "insights": "2-3 observaciones clave sobre los datos en español"
}}"""

        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response, handling potential JSON issues
        response_text = response.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        result = json.loads(response_text)
        
        # Validate the response
        if 'visualization_type' not in result:
            raise ValueError("Missing visualization_type in response")
        
        if result['visualization_type'] not in VISUALIZATION_TYPES:
            result['visualization_type'] = 'bar'  # Fallback
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"AI JSON Parse Error: {e}")
        return create_fallback_suggestion(df)
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return create_fallback_suggestion(df)


def create_fallback_suggestion(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create intelligent fallback suggestion when AI fails.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Determine best visualization based on data structure
    if len(numeric_cols) >= 2:
        viz_type = 'scatter'
        reasoning = 'Con múltiples columnas numéricas, un diagrama de dispersión permite ver relaciones.'
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
    elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
        viz_type = 'bar'
        reasoning = 'Una columna numérica con categorías es ideal para un gráfico de barras.'
        x_col = categorical_cols[0]
        y_col = numeric_cols[0]
    elif len(numeric_cols) == 1:
        viz_type = 'histogram'
        reasoning = 'Para analizar la distribución de una variable numérica, el histograma es ideal.'
        x_col = numeric_cols[0]
        y_col = None
    elif len(categorical_cols) >= 1:
        viz_type = 'bar'
        reasoning = 'Para datos categóricos, un gráfico de barras muestra las frecuencias.'
        x_col = categorical_cols[0]
        y_col = None
    else:
        viz_type = 'bar'
        reasoning = 'Visualización predeterminada basada en la estructura de datos.'
        x_col = df.columns[0] if len(df.columns) > 0 else None
        y_col = df.columns[1] if len(df.columns) > 1 else None
    
    return {
        'visualization_type': viz_type,
        'reasoning': reasoning,
        'x_column': x_col,
        'y_column': y_col,
        'color_column': None,
        'insights': 'Análisis automático basado en la estructura de tus datos.'
    }


# =============================================================================
# Visualization Creation
# =============================================================================

def create_visualization(
    df: pd.DataFrame, 
    viz_type: str, 
    params: Optional[Dict] = None
) -> Optional[go.Figure]:
    """
    Create visualization based on type and parameters.
    """
    try:
        fig = None
        template = "plotly_white"
        color_palette = px.colors.qualitative.Set2
        
        # Clean data for visualization
        df_clean = df.copy()
        
        # Get column info
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Extract params
        x_col = params.get('x_column') if params else None
        y_col = params.get('y_column') if params else None
        color_col = params.get('color_column') if params else None
        
        # Validate columns exist
        if x_col and x_col not in df_clean.columns:
            x_col = None
        if y_col and y_col not in df_clean.columns:
            y_col = None
        if color_col and color_col not in df_clean.columns:
            color_col = None

        # Create visualization based on type
        if viz_type == 'bar':
            fig = create_bar_chart(df_clean, x_col, y_col, color_col, 
                                   numeric_cols, categorical_cols, template, color_palette)
        
        elif viz_type == 'line':
            fig = create_line_chart(df_clean, x_col, y_col, color_col,
                                    numeric_cols, template, color_palette)
        
        elif viz_type == 'scatter':
            fig = create_scatter_chart(df_clean, x_col, y_col, color_col,
                                       numeric_cols, template, color_palette)
        
        elif viz_type == 'pie':
            fig = create_pie_chart(df_clean, y_col, x_col,
                                   numeric_cols, categorical_cols, template, color_palette)
        
        elif viz_type == 'heatmap':
            fig = create_heatmap(df_clean, numeric_cols, template)
        
        elif viz_type == 'box':
            fig = create_box_chart(df_clean, x_col, y_col,
                                   numeric_cols, categorical_cols, template, color_palette)
        
        elif viz_type == 'histogram':
            fig = create_histogram(df_clean, x_col, numeric_cols, template, color_palette)
        
        elif viz_type == 'area':
            fig = create_area_chart(df_clean, x_col, y_col, numeric_cols, template, color_palette)
        
        elif viz_type == 'bubble':
            fig = create_bubble_chart(df_clean, numeric_cols, template, color_palette)
        
        elif viz_type == 'treemap':
            fig = create_treemap(df_clean, categorical_cols, numeric_cols, template, color_palette)
        
        elif viz_type == 'sunburst':
            fig = create_sunburst(df_clean, categorical_cols, numeric_cols, template, color_palette)
        
        elif viz_type == 'violin':
            fig = create_violin_chart(df_clean, x_col, y_col, 
                                      numeric_cols, categorical_cols, template, color_palette)
        
        elif viz_type == 'radar':
            fig = create_radar_chart(df_clean, numeric_cols, template)
        
        elif viz_type == 'waterfall':
            fig = create_waterfall_chart(df_clean, x_col, y_col,
                                         numeric_cols, categorical_cols, template)
        
        elif viz_type == 'funnel':
            fig = create_funnel_chart(df_clean, categorical_cols, numeric_cols, 
                                      template, color_palette)
        
        # Fallback to bar chart
        if fig is None:
            fig = create_bar_chart(df_clean, None, None, None,
                                   numeric_cols, categorical_cols, template, color_palette)
        
        # Apply common styling
        if fig:
            fig.update_layout(
                font=dict(family="Inter, 'Segoe UI', sans-serif", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(bgcolor="white", font_size=13),
                margin=dict(t=50, b=50, l=50, r=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        
        return fig
        
    except Exception as e:
        print(f"Visualization Error: {e}")
        return None


# Individual chart creation functions
def create_bar_chart(df, x_col, y_col, color_col, numeric_cols, categorical_cols, template, colors):
    """Create bar chart."""
    if x_col and y_col:
        return px.bar(df, x=x_col, y=y_col, color=color_col,
                      template=template, color_discrete_sequence=colors)
    elif categorical_cols and numeric_cols:
        return px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                      template=template, color_discrete_sequence=colors)
    elif numeric_cols:
        return px.bar(df, y=numeric_cols[0], template=template,
                      color_discrete_sequence=colors)
    elif categorical_cols:
        counts = df[categorical_cols[0]].value_counts().reset_index()
        counts.columns = ['category', 'count']
        return px.bar(counts, x='category', y='count',
                      template=template, color_discrete_sequence=colors)
    return None


def create_line_chart(df, x_col, y_col, color_col, numeric_cols, template, colors):
    """Create line chart."""
    if x_col and y_col:
        return px.line(df, x=x_col, y=y_col, color=color_col,
                       template=template, color_discrete_sequence=colors)
    elif numeric_cols:
        return px.line(df, y=numeric_cols[0], template=template,
                       color_discrete_sequence=colors)
    return None


def create_scatter_chart(df, x_col, y_col, color_col, numeric_cols, template, colors):
    """Create scatter plot."""
    if x_col and y_col:
        return px.scatter(df, x=x_col, y=y_col, color=color_col,
                          template=template, color_discrete_sequence=colors)
    elif len(numeric_cols) >= 2:
        return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                          template=template, color_discrete_sequence=colors)
    return None


def create_pie_chart(df, values_col, names_col, numeric_cols, categorical_cols, template, colors):
    """Create pie chart."""
    if values_col and names_col:
        return px.pie(df, values=values_col, names=names_col,
                      template=template, color_discrete_sequence=colors)
    elif numeric_cols and categorical_cols:
        # Aggregate if needed
        agg_df = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().reset_index()
        return px.pie(agg_df, values=numeric_cols[0], names=categorical_cols[0],
                      template=template, color_discrete_sequence=colors)
    elif categorical_cols:
        counts = df[categorical_cols[0]].value_counts().reset_index()
        counts.columns = ['category', 'count']
        return px.pie(counts, values='count', names='category',
                      template=template, color_discrete_sequence=colors)
    return None


def create_heatmap(df, numeric_cols, template):
    """Create correlation heatmap."""
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        return px.imshow(corr, template=template, color_continuous_scale='RdBu',
                         text_auto='.2f', aspect='auto')
    return None


def create_box_chart(df, x_col, y_col, numeric_cols, categorical_cols, template, colors):
    """Create box plot."""
    if x_col and y_col:
        return px.box(df, x=x_col, y=y_col, template=template,
                      color_discrete_sequence=colors)
    elif categorical_cols and numeric_cols:
        return px.box(df, x=categorical_cols[0], y=numeric_cols[0],
                      template=template, color_discrete_sequence=colors)
    elif numeric_cols:
        return px.box(df, y=numeric_cols[0], template=template,
                      color_discrete_sequence=colors)
    return None


def create_histogram(df, x_col, numeric_cols, template, colors):
    """Create histogram."""
    col = x_col if x_col and x_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    if col:
        return px.histogram(df, x=col, template=template, color_discrete_sequence=colors)
    return None


def create_area_chart(df, x_col, y_col, numeric_cols, template, colors):
    """Create area chart."""
    if x_col and y_col:
        return px.area(df, x=x_col, y=y_col, template=template,
                       color_discrete_sequence=colors)
    elif numeric_cols:
        return px.area(df, y=numeric_cols[0], template=template,
                       color_discrete_sequence=colors)
    return None


def create_bubble_chart(df, numeric_cols, template, colors):
    """Create bubble chart."""
    if len(numeric_cols) >= 3:
        return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                          size=numeric_cols[2], template=template,
                          color_discrete_sequence=colors)
    return None


def create_treemap(df, categorical_cols, numeric_cols, template, colors):
    """Create treemap."""
    if categorical_cols and numeric_cols:
        return px.treemap(df, path=[categorical_cols[0]], values=numeric_cols[0],
                          template=template, color_discrete_sequence=colors)
    return None


def create_sunburst(df, categorical_cols, numeric_cols, template, colors):
    """Create sunburst chart."""
    if categorical_cols and numeric_cols:
        return px.sunburst(df, path=[categorical_cols[0]], values=numeric_cols[0],
                           template=template, color_discrete_sequence=colors)
    return None


def create_violin_chart(df, x_col, y_col, numeric_cols, categorical_cols, template, colors):
    """Create violin plot."""
    if x_col and y_col:
        return px.violin(df, x=x_col, y=y_col, template=template,
                         color_discrete_sequence=colors, box=True)
    elif categorical_cols and numeric_cols:
        return px.violin(df, x=categorical_cols[0], y=numeric_cols[0],
                         template=template, color_discrete_sequence=colors, box=True)
    elif numeric_cols:
        return px.violin(df, y=numeric_cols[0], template=template,
                         color_discrete_sequence=colors, box=True)
    return None


def create_radar_chart(df, numeric_cols, template):
    """Create radar/spider chart."""
    if len(numeric_cols) >= 3:
        cols = numeric_cols[:6]  # Limit to 6 variables
        r = df[cols].mean().values.tolist()
        r.append(r[0])  # Close the polygon
        theta = list(cols) + [cols[0]]
        
        fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta, fill='toself'))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(r) * 1.1])),
            showlegend=False, 
            template=template
        )
        return fig
    return None


def create_waterfall_chart(df, x_col, y_col, numeric_cols, categorical_cols, template):
    """Create waterfall chart."""
    if categorical_cols and numeric_cols:
        x_vals = df[categorical_cols[0]].tolist()
        y_vals = df[numeric_cols[0]].tolist()
    elif numeric_cols:
        x_vals = list(range(len(df)))
        y_vals = df[numeric_cols[0]].tolist()
    else:
        return None
    
    fig = go.Figure(go.Waterfall(
        x=x_vals,
        y=y_vals,
        connector={"line": {"color": "rgb(63, 63, 63)"}}
    ))
    fig.update_layout(template=template)
    return fig


def create_funnel_chart(df, categorical_cols, numeric_cols, template, colors):
    """Create funnel chart."""
    if categorical_cols and numeric_cols:
        return px.funnel(df, x=numeric_cols[0], y=categorical_cols[0],
                         template=template, color_discrete_sequence=colors)
    return None


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial analysis."""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró archivo'}), 400
        
        file = request.files['file']
        context = request.form.get('context', '')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'}), 400
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return jsonify({
                'success': False, 
                'error': f'Formato no soportado. Usa: {", ".join(app.config["UPLOAD_EXTENSIONS"])}'
            }), 400
        
        # Read and validate file
        df, error_msg = read_data_file(file, file_ext)
        
        if df is None:
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Limit data size for analysis
        df_for_analysis = df.head(app.config['MAX_ROWS_FOR_ANALYSIS'])
        
        # Get AI analysis
        ai_suggestion = analyze_data_with_ai(df_for_analysis, context)
        
        # Create initial visualization
        initial_fig = create_visualization(df, ai_suggestion['visualization_type'], ai_suggestion)
        
        # Prepare response
        response_data = {
            'success': True,
            'data': df.to_json(orient='records'),
            'columns': list(df.columns),
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'ai_suggestion': ai_suggestion,
            'initial_chart': initial_fig.to_json() if initial_fig else None,
            'visualization_types': VISUALIZATION_TYPES
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Error al procesar el archivo: {str(e)}'
        }), 500


@app.route('/visualize', methods=['POST'])
def visualize():
    """Create visualization for given data and type."""
    try:
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'Datos no proporcionados'}), 400
        
        df = pd.DataFrame(data['data'])
        viz_type = data.get('type', 'bar')
        params = data.get('params', {})
        
        if viz_type not in VISUALIZATION_TYPES:
            viz_type = 'bar'
        
        fig = create_visualization(df, viz_type, params)
        
        if fig:
            return jsonify({
                'success': True,
                'chart': fig.to_json()
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'No se pudo crear la visualización con estos datos'
            }), 400
            
    except Exception as e:
        print(f"Visualize Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/export', methods=['POST'])
def export_chart():
    """Export chart to image format."""
    try:
        data = request.json
        chart_json = data.get('chart')
        format_type = data.get('format', 'png')
        
        if not chart_json:
            return jsonify({'success': False, 'error': 'No hay gráfico para exportar'}), 400
        
        if format_type not in ['png', 'jpg', 'jpeg', 'pdf']:
            return jsonify({'success': False, 'error': 'Formato no soportado'}), 400
        
        # Recreate figure from JSON
        fig = pio.from_json(chart_json)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as tmp_file:
            img_format = 'jpeg' if format_type in ['jpg', 'jpeg'] else format_type
            fig.write_image(tmp_file.name, format=img_format, scale=2)
            tmp_file.flush()
            
            # Read and encode
            with open(tmp_file.name, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            
            # Cleanup
            os.unlink(tmp_file.name)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return jsonify({
                'success': True,
                'data': b64,
                'filename': f'curva_io_chart_{timestamp}.{format_type}'
            })
            
    except Exception as e:
        print(f"Export Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/sheets', methods=['POST'])
def get_excel_sheets():
    """Get list of sheets from Excel file (for potential future use)."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        excel_file = pd.ExcelFile(file)
        
        return jsonify({
            'success': True,
            'sheets': excel_file.sheet_names
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
