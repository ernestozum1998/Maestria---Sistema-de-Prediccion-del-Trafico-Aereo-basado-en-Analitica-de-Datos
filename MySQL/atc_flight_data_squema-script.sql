-- Usar la base de datos
USE atc_flight_data;

-- Tabla de aeropuertos (sin duplicados)
CREATE TABLE IF NOT EXISTS airports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    icao_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100)
);

-- Tabla de aeropuertos (sin duplicados)
CREATE TABLE IF NOT EXISTS airlines (
    icao_prefix VARCHAR(20) UNIQUE NOT NULL PRIMARY KEY,
    airline_name VARCHAR(100),
    flight_count int 
);

-- Tabla para metadatos de archivos procesados
CREATE TABLE IF NOT EXISTS metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255),
    date_loaded DATETIME DEFAULT CURRENT_TIMESTAMP,
    fpl_count INT
);

-- Tabla principal de mensajes FPL
CREATE TABLE IF NOT EXISTS flight_plans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    callsign VARCHAR(20) NOT NULL,
    departure_airport_id INT,
    arrival_airport_id INT,
    departure_time VARCHAR(5),
    arrival_time VARCHAR(5),
    raw_route TEXT,
	metadata_id INT,
    FOREIGN KEY (departure_airport_id) REFERENCES airports(id),
    FOREIGN KEY (arrival_airport_id) REFERENCES airports(id),
	FOREIGN KEY (metadata_id) REFERENCES metadata(id)
);

-- Tabla de puntos de ruta (waypoints)
CREATE TABLE IF NOT EXISTS waypoints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fpl_message_id INT,
    sequence INT,
    name VARCHAR(50),
    FOREIGN KEY (fpl_message_id) REFERENCES flight_plans(id)
);

-- 1. Agregar columnas a la tabla flight_plans
ALTER TABLE flight_plans
ADD COLUMN fecha DATE,
ADD COLUMN anio INT,
ADD COLUMN mes INT,
ADD COLUMN dia INT;

-- 2. Poblar las nuevas columnas usando el campo file_name en metadata
UPDATE flight_plans fp
JOIN metadata m ON fp.metadata_id = m.id
SET
  fp.fecha = STR_TO_DATE(
                SUBSTRING_INDEX(SUBSTRING_INDEX(m.file_name, '.', -3), '.', 3),
                '%Y.%m.%d'
            ),
  fp.anio = YEAR(STR_TO_DATE(
                SUBSTRING_INDEX(SUBSTRING_INDEX(m.file_name, '.', -3), '.', 3),
                '%Y.%m.%d'
            )),
  fp.mes = MONTH(STR_TO_DATE(
                SUBSTRING_INDEX(SUBSTRING_INDEX(m.file_name, '.', -3), '.', 3),
                '%Y.%m.%d'
            )),
  fp.dia = DAY(STR_TO_DATE(
                SUBSTRING_INDEX(SUBSTRING_INDEX(m.file_name, '.', -3), '.', 3),
                '%Y.%m.%d'
            ));

-- 3. Crear índices para optimizar consultas por fecha
CREATE INDEX idx_fecha ON flight_plans(fecha);
CREATE INDEX idx_anio ON flight_plans(anio);
CREATE INDEX idx_mes ON flight_plans(mes);
CREATE INDEX idx_dia ON flight_plans(dia);

CREATE OR REPLACE VIEW vuelos_por_dia AS
SELECT 
    anio,
    mes,
    dia,
    COUNT(*) AS total_vuelos
FROM flight_plans
GROUP BY anio, mes, dia
ORDER BY anio, mes, dia;

CREATE OR REPLACE VIEW vuelos_por_mes AS
SELECT 
    anio,
    mes,
    COUNT(*) AS total_vuelos
FROM flight_plans
GROUP BY anio, mes
ORDER BY anio, mes;

CREATE OR REPLACE VIEW vuelos_por_anio AS
SELECT 
    anio,
    COUNT(*) AS total_vuelos
FROM flight_plans
GROUP BY anio
ORDER BY anio;

CREATE OR REPLACE VIEW vuelos_por_hora AS
SELECT
    anio,
    mes,
    dia,
    LPAD(FLOOR(departure_time / 100), 2, '0') AS hora,
    COUNT(*) AS total_vuelos
FROM flight_plans
WHERE departure_time IS NOT NULL
GROUP BY
    anio, mes, dia, hora
ORDER BY
    anio, mes, dia, hora;
    
