-- Procedure for add a new correction for a student
DELIMITER $$
DROP PROCEDURE IF EXISTS AddBonus;
CREATE PROCEDURE  AddBonus(
    user_id INT,
    project_name VARCHAR(255),
    score INT
)
BEGIN
    SET @number =( SELECT COUNT(*) FROM projects WHERE projects.name LIKE project_name);
    IF @number = 0 THEN
        INSERT INTO projects(name)
            VALUES(project_name);
    END IF;

    INSERT INTO corrections
    VALUES(user_id, (SELECT id FROM projects WHERE name = project_name), score);
END$$
DELIMITER ;
