<!DOCTYPE html>
<html>
<head>
    <title>Student Registration Form</title>
</head>
<body>

<h2>Student Registration Form</h2>

<form action="action.php" method="POST">

    <label>Student Name:</label><br>
    <input type="text" name="name" required><br><br>

    <label>Email ID:</label><br>
    <input type="email" name="email" required><br><br>

    <label>Phone Number:</label><br>
    <input type="text" name="phone" required><br><br>

    <label>Course Name:</label><br>
    <input type="text" name="course" required><br><br>

    <label>Semester:</label><br>
    <input type="number" name="semester" required><br><br>

    <label>Address:</label><br>
    <textarea name="address" required></textarea><br><br>

    <input type="submit" value="Submit">
    <input type="reset" value="Reset">

</form>

</body>
</html>