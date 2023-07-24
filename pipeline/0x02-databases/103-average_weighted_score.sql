--  script that creates a stored procedure ComputeAverageWeightedScoreForUser
-- that computes and store the average weighted score for a student
DELIMITER $$
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN user_id_new INTEGER)
BEGIN
    UPDATE users
    SET average_score = (
        SELECT SUM(score * weight) / SUM(weight)
        FROM corrections c
        JOIN projects p
        ON c.project_id=p.id
        WHERE user_id=user_id_new
    )
    WHERE id=user_id_new;
END$$
DELIMITER ;
