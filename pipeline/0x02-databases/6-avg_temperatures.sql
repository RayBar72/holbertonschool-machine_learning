-- Script that displays average temperature by city orderer by temperature
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
