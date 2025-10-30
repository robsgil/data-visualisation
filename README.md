# Curva.io - Visualización de Datos con IA


## 📊 Descripción

Curva.io es una plataforma web inteligente para la visualización de datos que utiliza inteligencia artificial (Claude Sonnet) para analizar tus datos y sugerir la mejor forma de visualizarlos. La plataforma está diseñada con un enfoque en la democratización y accesibilidad de las herramientas de análisis de datos.

## ✨ Características

- **🤖 IA Inteligente**: Análisis automático de datos y sugerencias de visualización usando Claude
- **🔒 100% Privado**: No guardamos ningún dato - todo el procesamiento es en tiempo real
- **📱 Responsive**: Diseño adaptativo que funciona en móvil, tablet y desktop
- **📈 15+ Tipos de Gráficos**: Bar, línea, dispersión, circular, heatmap, y muchos más
- **💾 Exportación Múltiple**: Descarga tus gráficos en PNG, JPG o PDF
- **🎨 Diseño Moderno**: Interfaz atractiva con animaciones suaves
- **🌐 En Español**: Interfaz completamente en español

## 🚀 Instalación

### Prerrequisitos

- Python 3.8+
- pip
- Node.js (opcional, para desarrollo frontend)

### Configuración Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/curva-io.git
cd curva-io
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**

Crear un archivo `.env` en la raíz del proyecto:
```env
CLAUDE_API_KEY=tu-api-key-de-anthropic
PORT=5000
```

5. **Ejecutar la aplicación**
```bash
python app.py
```

La aplicación estará disponible en `http://localhost:5000`

## 🚢 Despliegue en Railway

1. **Fork este repositorio** a tu cuenta de GitHub

2. **Conecta con Railway**
   - Ve a [Railway.app](https://railway.app)
   - Crea un nuevo proyecto
   - Conecta tu repositorio de GitHub

3. **Configura las variables de entorno**
   - En Railway, ve a Variables
   - Agrega `CLAUDE_API_KEY` con tu clave de API de Anthropic
   - Agrega `PORT` (Railway lo configura automáticamente)

4. **Deploy**
   - Railway detectará automáticamente que es una aplicación Python
   - El despliegue se iniciará automáticamente
   - Tu app estará disponible en la URL proporcionada por Railway

## 📁 Estructura del Proyecto

```
curva_io/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias de Python
├── templates/
│   └── index.html        # Template HTML principal
├── static/
│   ├── css/
│   │   └── styles.css    # Estilos CSS
│   └── js/
│       └── main.js       # JavaScript frontend
└── README.md             # Este archivo
```

## 🔧 Configuración de la API de Claude

1. Ve a [Anthropic Console](https://console.anthropic.com/)
2. Crea una cuenta si no tienes una
3. Genera una API key
4. Agrega la key a tu archivo `.env` o variables de entorno

## 💻 Uso

1. **Cargar datos**: Arrastra o selecciona un archivo CSV o Excel
2. **Agregar contexto**: Describe tus datos para mejores sugerencias (opcional)
3. **Procesar con IA**: La IA analizará y sugerirá la mejor visualización
4. **Personalizar**: Cambia el tipo de gráfico si lo deseas
5. **Descargar**: Exporta tu gráfico en el formato preferido

## 🛠️ Tecnologías Utilizadas

### Backend
- **Flask**: Framework web de Python
- **Pandas**: Procesamiento de datos
- **Plotly**: Generación de gráficos interactivos
- **Anthropic Claude**: Análisis inteligente de datos

### Frontend
- **HTML5/CSS3**: Estructura y estilos
- **JavaScript**: Interactividad
- **Plotly.js**: Renderizado de gráficos en el navegador
- **Font Awesome**: Iconos

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- Anthropic por la API de Claude
- La comunidad open source
- Todos los usuarios que confían en Curva.io para sus visualizaciones

## 📧 Contacto

Para preguntas, sugerencias o reportar bugs, por favor abre un issue en GitHub.

---

**Hecho con ❤️ para democratizar la visualización de datos**
