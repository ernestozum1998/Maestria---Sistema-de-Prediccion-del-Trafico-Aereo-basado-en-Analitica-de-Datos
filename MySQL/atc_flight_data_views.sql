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