-- Script that creates a trigger that resets the attribute
-- valid_emai only when the email has been changed
delimiter //
DROP TRIGGER IF EXISTS email_valid;
CREATE TRIGGER email_valid
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.email <> OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END;
//
delimiter ;
