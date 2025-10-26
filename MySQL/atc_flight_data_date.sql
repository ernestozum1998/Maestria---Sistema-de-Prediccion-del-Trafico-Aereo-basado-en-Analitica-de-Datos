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
