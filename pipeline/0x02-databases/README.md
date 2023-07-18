# 0x02. Databases #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- General
- What’s a relational database
- What’s a none relational database
- What is difference between SQL and NoSQL
- How to create tables with constraints
- How to optimize queries by adding indexes
- What is and how to implement stored procedures and functions in MySQL
- What is and how to implement views in MySQL
- What is and how to implement triggers in MySQL
- What is ACID
- What is a document storage
- What are NoSQL types
- What are benefits of a NoSQL database
- How to query information from a NoSQL database
- How to insert/update/delete information from a NoSQL database
- How to use MongoDB
- Requirements

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Create a database | Write a script that creates the database db_0 in your MySQL server. | 0-create_database_if_missing.sql |
| 1. First table | Write a script that creates a table called first_table in the current database in your MySQL server. | 1-first_table.sql |
| 2. List all in table | Write a script that lists all rows of the table first_table in your MySQL server. | 2-list_values.sql |
| 3. First add | Write a script that inserts a new row in the table first_table in your MySQL server. | 3-insert_value.sql |
| 4. Select the best | Write a script that lists all records with a score >= 10 in the table second_table in your MySQL server. | 4-best_score.sql |
| 5. Average | Write a script that computes the score average of all records in the table second_table in your MySQL server. | 5-average.sql |
| 6. Temperatures #0 | Write a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending). | 6-avg_temperatures.sql |
| 7. Temperatures #2 | Write a script that displays the max temperature of each state (ordered by State name). | 7-max_state.sql |
| 8. Genre ID by show | Write a script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked. | 8-genre_id_by_show.sql |
| 9. No genre | Write a script that lists all shows contained in hbtn_0d_tvshows without a genre linked. | 9-no_genre.sql |
| 10. Number of shows by genre | Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each. | 10-count_shows_by_genre.sql |
| 11. Rotten tomatoes | Write a script that lists all shows from hbtn_0d_tvshows_rate by their rating. | 11-rating_shows.sql |
| 12. Best genre | Write a script that lists all genres in the database hbtn_0d_tvshows_rate by their rating. | 12-rating_genres.sql |
| 13. We are all unique! | Write a SQL script that creates a table users following these requirements | 13-uniq_users.sql |
| 14. In and not out | Write a SQL script that creates a table users following these requirements | 14-country_users.sql |
| 15. Best band ever! | Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans | 15-fans.sql |
| 16. Old school band | Write a SQL script that lists all bands with Glam rock as their main style, ranked by their longevity | 16-glam_rock.sql |
| 17. Buy buy buy | Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order. | 17-store.sql |
| 18. Email validation to sent | Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed. | 18-valid_email.sql |
| 19. Add bonus | Write a SQL script that creates a stored procedure AddBonus that adds a new correction for a student. | 19-bonus.sql |
| 20. Average score | Write a SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student. | 20-average_score.sql |
| 21. Safe divide | Write a SQL script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0. | 21-div.sql |
| 22. List all databases | Write a script that lists all databases in MongoDB. | 22-list_databases |
| 23. Create a database | Write a script that creates or uses the database my_db | 23-use_or_create_database |
| 24. Insert document | Write a script that inserts a document in the collection school | 24-insert |
| 25. All documents | Write a script that lists all documents in the collection school | 25-all |
| 26. All matches | Write a script that lists all documents with name="Holberton school" in the collection school | 26-match |
| 27. Count | Write a script that displays the number of documents in the collection school | 27-count |
| 28. Update | Write a script that adds a new attribute to a document in the collection school | 28-update |
| 29. Delete by match | Write a script that deletes all documents with name="Holberton school" in the collection school | 29-delete |
| 30. List all documents in Python | Write a Python function that lists all documents in a collection | 30-all.py |
| 31. Insert a document in Python | Write a Python function that inserts a new document in a collection based on kwargs | 31-insert_school.py |
| 32. Change school topics | Write a Python function that changes all topics of a school document based on the name | 32-update_topics.py |
| 33. Where can I learn Python? | Write a Python function that returns the list of school having a specific topic | 33-schools_by_topic.py |
| 34. Log stats | Write a Python script that provides some stats about Nginx logs stored in MongoDB | 34-log_stats.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/


**Project Required by**: HolbertonSchool
