# Usa una imagen base de Python
FROM python:3.9-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos necesarios
COPY main.py .
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto donde corre la aplicación FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI dentro del contenedor
CMD ["uvicorn", "main:app", "--port", "8000"]
