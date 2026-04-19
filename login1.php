<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
</head>
<body>

<form action="" method="POST">

    <input type="email" name="email" placeholder="Email ID" required><br><br>

    <input type="password" name="password" placeholder="Password" required><br><br>

    <input type="submit" name="login" value="Login">
    <input type="reset" value="Reset">

</form>

<?php

$host = "localhost";
$username = "root";
$password = "manager";
$database = "student_db";

// Create connection
$conn = new mysqli($host, $username, $password);

// Check connection
if ($conn->connect_error) {
    die();
}

// Select database (FIX)
$conn->select_db($database);

// Login logic
if (isset($_POST['login'])) {

    $email = $_POST['email'];
    $pass = $_POST['password'];

    $sql = "SELECT * FROM log WHERE email_id='$email' AND password='$pass'";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // success (no output)
    } else {
        // failure (no output)
    }
}

$conn->close();

?>

</body>
</html>