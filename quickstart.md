# 🚀 Guía de Inicio Rápido - Curva.io

## Opción 1: Ejecutar Localmente (5 minutos)

### 1. Requisitos Previos
- Python 3.8 o superior instalado
- pip (gestor de paquetes de Python)

### 2. Instalación Rápida

```bash
# Clonar o descargar el proyecto
cd curva_io

# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key de Claude
export CLAUDE_API_KEY="tu-api-key-aqui"  # Linux/Mac
set CLAUDE_API_KEY="tu-api-key-aqui"     # Windows
```

### 3. Ejecutar la Aplicación

```bash
python app.py
```

Abre tu navegador en: `http://localhost:5000`

### 4. Probar la Aplicación
- Usa el archivo `sample_data.csv` incluido para una prueba rápida
- O sube tu propio archivo CSV/Excel
- Agrega contexto como: "Datos de ventas mensuales, quiero ver tendencias"

## Opción 2: Desplegar en Railway (10 minutos)

### 1. Preparación
1. Fork o sube este proyecto a tu GitHub
2. Obtén tu API Key de Anthropic desde [console.anthropic.com](https://console.anthropic.com)

### 2. Despliegue
1. Ve a [railway.app](https://railway.app)
2. Haz clic en "New Project" → "Deploy from GitHub"
3. Selecciona tu repositorio
4. Railway detectará automáticamente que es una app Python

### 3. Configuración
En Railway, ve a Variables y agrega:
- `CLAUDE_API_KEY` = tu-api-key-de-anthropic
- Railway configurará `PORT` automáticamente

### 4. ¡Listo!
Railway te proporcionará una URL pública para tu aplicación

## 🎯 Características Principales

### ✅ Lo que SÍ hace Curva.io:
- Analiza automáticamente tus datos con IA
- Sugiere la mejor visualización basada en contexto
- Ofrece 15+ tipos de gráficos diferentes
- Exporta en PNG, JPG y PDF
- Funciona en móvil, tablet y desktop
- 100% privado - no guarda datos

### ❌ Lo que NO hace (por diseño):
- No requiere registro de usuarios
- No guarda datos después del procesamiento
- No requiere pago o suscripción
- No tiene límites de uso

## 📊 Tipos de Datos Soportados

### Archivos:
- CSV (.csv)
- Excel (.xlsx, .xls)
- Máximo 16MB por archivo

### Mejores Prácticas:
1. **Estructura de datos**: Primera fila con nombres de columnas
2. **Contexto**: Mientras más específico, mejor la sugerencia
3. **Tamaño**: Para mejor rendimiento, <1000 filas

## 🔧 Solución de Problemas

### Error: "No module named..."
```bash
pip install -r requirements.txt
```

### Error: API Key inválida
- Verifica tu CLAUDE_API_KEY en las variables de entorno
- Asegúrate de que la key esté activa en console.anthropic.com

### Gráfico no se visualiza
- Verifica que tu navegador tenga JavaScript habilitado
- Prueba con un archivo más pequeño primero

## 📚 Recursos Adicionales

- [Documentación de Plotly](https://plotly.com/python/)
- [API de Anthropic](https://docs.anthropic.com)
- [Ejemplos de visualizaciones](https://plotly.com/python/basic-charts/)

## 💡 Tips para Mejores Resultados

1. **Contexto específico**: 
   - ❌ "mis datos"
   - ✅ "ventas mensuales del 2024, comparar trimestres"

2. **Datos limpios**:
   - Elimina filas vacías
   - Usa nombres de columnas descriptivos
   - Formatea fechas consistentemente

3. **Explora diferentes visualizaciones**:
   - La IA sugiere una, pero puedes probar todas
   - Algunos datos se ven mejor en ciertos gráficos

## 🤝 ¿Necesitas Ayuda?

- 📧 Abre un issue en GitHub
- 💬 Revisa los issues existentes
- 📖 Lee la documentación completa en README.md

---

**¡Disfruta visualizando tus datos con Curva.io!** 🎉
