-- Function that makes a safe divison

DELIMITER $$
DROP FUNCTION IF EXISTS SafeDiv;
CREATE FUNCTION SafeDiv(
    a INT,
    b INT
)
RETURNS FLOAT
BEGIN
    RETURN IF(b = 0, 0, a /b);
END$$

DELIMITER ;
