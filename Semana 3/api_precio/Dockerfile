# Usa la imagen base de Python
FROM python:3.8

# Establece el directorio de trabajo en /app
WORKDIR /api_precio

# Copia el archivo de requisitos (requirements.txt) al contenedor
COPY proyecto/ .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido del directorio actual al directorio /app en el contenedor

# Exponer el puerto 5000, que es donde se ejecutará la API
EXPOSE 5000

# Comando para ejecutar la API
CMD ["python", "app.py"]
