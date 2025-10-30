import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import base64
from anthropic import Anthropic
import plotly.io as pio
from datetime import datetime
import tempfile

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.xls']

# Placeholder for Claude API key
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', 'your-api-key-here')

# Initialize Anthropic client
anthropic = Anthropic(api_key=CLAUDE_API_KEY)

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
    'funnel': 'Gráfico de Embudo',
    'choropleth': 'Mapa Coroplético'
}

def analyze_data_with_ai(df, context):
    """Use Claude to analyze data and suggest best visualization"""
    try:
        # Prepare data summary for Claude
        data_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'sample': df.head(5).to_dict(),
            'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
            'context': context
        }
        
        prompt = f"""
        Analyze this dataset and suggest the best visualization type.
        
        Data Information:
        - Columns: {data_info['columns']}
        - Data types: {data_info['dtypes']}
        - Shape: {data_info['shape']}
        - Numeric columns: {data_info['numeric_cols']}
        - Categorical columns: {data_info['categorical_cols']}
        
        User Context: {context}
        
        Based on the data structure and user context, respond with a JSON object containing:
        1. "visualization_type": one of [bar, line, scatter, pie, heatmap, box, histogram, area, bubble, treemap, sunburst, violin, radar, waterfall, funnel]
        2. "reasoning": explanation in Spanish why this visualization is best
        3. "x_column": suggested column for x-axis (if applicable)
        4. "y_column": suggested column for y-axis (if applicable)
        5. "color_column": suggested column for color grouping (if applicable)
        6. "insights": key insights about the data in Spanish
        
        Respond ONLY with the JSON object, no additional text.
        """
        
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        result = json.loads(response.content[0].text)
        return result
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        # Fallback logic
        return {
            'visualization_type': 'bar' if len(df.columns) > 1 else 'histogram',
            'reasoning': 'Análisis automático basado en la estructura de datos.',
            'x_column': df.columns[0] if len(df.columns) > 0 else None,
            'y_column': df.columns[1] if len(df.columns) > 1 else None,
            'insights': 'Análisis básico de los datos proporcionados.'
        }

def create_visualization(df, viz_type, params=None):
    """Create visualization based on type and parameters"""
    try:
        fig = None
        
        # Set default theme
        template = "plotly_white"
        color_palette = px.colors.qualitative.Set3
        
        if viz_type == 'bar':
            if params and params.get('x_column') and params.get('y_column'):
                fig = px.bar(df, x=params['x_column'], y=params['y_column'],
                           color=params.get('color_column'),
                           template=template, color_discrete_sequence=color_palette)
            else:
                # Auto-detect columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.bar(df, y=numeric_cols[0], template=template)
                    
        elif viz_type == 'line':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 1:
                fig = px.line(df, y=numeric_cols[0], template=template,
                            color_discrete_sequence=color_palette)
                
        elif viz_type == 'scatter':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               template=template, color_discrete_sequence=color_palette)
                
        elif viz_type == 'pie':
            if params and params.get('values_column'):
                fig = px.pie(df, values=params['values_column'],
                           names=params.get('names_column', df.columns[0]),
                           template=template, color_discrete_sequence=color_palette)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.pie(df, values=numeric_cols[0], names=df.columns[0],
                               template=template)
                    
        elif viz_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig = px.imshow(numeric_df.corr(), template=template,
                              color_continuous_scale='RdBu')
                
        elif viz_type == 'box':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.box(df, y=numeric_cols[0], template=template,
                           color_discrete_sequence=color_palette)
                
        elif viz_type == 'histogram':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], template=template,
                                 color_discrete_sequence=color_palette)
                
        elif viz_type == 'area':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.area(df, y=numeric_cols[0], template=template,
                            color_discrete_sequence=color_palette)
                
        elif viz_type == 'bubble':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               size=numeric_cols[2], template=template,
                               color_discrete_sequence=color_palette)
                
        elif viz_type == 'treemap':
            if len(df.columns) >= 2:
                # Assuming first column is labels, second is values
                fig = px.treemap(df, path=[df.columns[0]], values=df.columns[1],
                               template=template, color_discrete_sequence=color_palette)
                
        elif viz_type == 'sunburst':
            if len(df.columns) >= 2:
                fig = px.sunburst(df, path=[df.columns[0]], values=df.columns[1],
                                template=template, color_discrete_sequence=color_palette)
                
        elif viz_type == 'violin':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.violin(df, y=numeric_cols[0], template=template,
                              color_discrete_sequence=color_palette)
                
        elif viz_type == 'radar':
            # Create a simple radar chart
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 variables
            if len(numeric_cols) > 2:
                theta = list(numeric_cols)
                r = df[numeric_cols].mean().values
                fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta, fill='toself'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                showlegend=False, template=template)
                
        elif viz_type == 'waterfall':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(df) > 0:
                values = df[numeric_cols[0]].values
                fig = go.Figure(go.Waterfall(
                    x=list(range(len(values))),
                    y=values,
                    connector={"line": {"color": "rgb(63, 63, 63)"}}
                ))
                fig.update_layout(template=template)
                
        elif viz_type == 'funnel':
            if len(df.columns) >= 2:
                fig = px.funnel(df, x=df.columns[1], y=df.columns[0],
                              template=template, color_discrete_sequence=color_palette)
        
        # Default to bar chart if nothing else works
        if fig is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.bar(df, y=numeric_cols[0], template=template)
            else:
                # Create a value counts bar chart for categorical data
                fig = px.bar(x=df[df.columns[0]].value_counts().index,
                           y=df[df.columns[0]].value_counts().values,
                           template=template)
        
        # Update layout with Curva.io styling
        if fig:
            fig.update_layout(
                font=dict(family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(bgcolor="white", font_size=14),
                margin=dict(t=40, b=40, l=40, r=40)
            )
        
        return fig
    except Exception as e:
        print(f"Visualization Error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró archivo'}), 400
        
        file = request.files['file']
        context = request.form.get('context', '')
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return jsonify({'error': 'Formato de archivo no soportado'}), 400
        
        # Read the file into pandas
        if file_ext == '.csv':
            df = pd.read_csv(file)
        else:  # Excel files
            df = pd.read_excel(file)
        
        # Store in session or temporary storage
        # For simplicity, we'll convert to JSON and send back
        data_json = df.to_json(orient='records')
        
        # Get AI analysis
        ai_suggestion = analyze_data_with_ai(df, context)
        
        # Create initial visualization
        initial_fig = create_visualization(df, ai_suggestion['visualization_type'], ai_suggestion)
        
        if initial_fig:
            initial_chart = initial_fig.to_json()
        else:
            initial_chart = None
        
        return jsonify({
            'success': True,
            'data': data_json,
            'columns': list(df.columns),
            'shape': df.shape,
            'ai_suggestion': ai_suggestion,
            'initial_chart': initial_chart,
            'visualization_types': VISUALIZATION_TYPES
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = request.json
        df = pd.DataFrame(data['data'])
        viz_type = data.get('type', 'bar')
        params = data.get('params', {})
        
        fig = create_visualization(df, viz_type, params)
        
        if fig:
            return jsonify({
                'success': True,
                'chart': fig.to_json()
            })
        else:
            return jsonify({'error': 'No se pudo crear la visualización'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_chart():
    try:
        data = request.json
        chart_json = data['chart']
        format_type = data.get('format', 'png')
        
        # Recreate figure from JSON
        fig = pio.from_json(chart_json)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as tmp_file:
            if format_type in ['png', 'jpg', 'jpeg']:
                fig.write_image(tmp_file.name, format='png' if format_type == 'png' else 'jpeg')
            elif format_type == 'pdf':
                fig.write_image(tmp_file.name, format='pdf')
            else:
                return jsonify({'error': 'Formato no soportado'}), 400
            
            tmp_file.flush()
            
            # Read the file and encode as base64
            with open(tmp_file.name, 'rb') as f:
                file_data = f.read()
                b64 = base64.b64encode(file_data).decode()
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return jsonify({
                'success': True,
                'data': b64,
                'filename': f'curva_io_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format_type}'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
