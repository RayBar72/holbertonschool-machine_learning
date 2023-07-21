-- Procedure that computes and store the average score for a student.
DELIMITER $$

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;
CREATE PROCEDURE ComputeAverageScoreForUser(
    user_id INT
)
BEGIN
    SET @aver = (SELECT AVG(score)
                FROM corrections c
                WHERE c.user_id = user_id);
    UPDATE users u
    SET average_score = @aver
    WHERE u.id = user_id;
END$$

DELIMITER ;
