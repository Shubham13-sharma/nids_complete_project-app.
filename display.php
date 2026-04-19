<?php

// 🔹 1. WRITING TO FILE (sample.txt)
$file1 = fopen("sample.txt", "w");

if ($file1) {
    fwrite($file1, "Welcome in the world of machine learning.");
    fclose($file1);
    echo "Write Successful<br>";
} else {
    echo "Error in writing file<br>";
}

// 🔹 2. APPENDING TO FILE (example.txt)
$file2 = fopen("example.txt", "a");

if ($file2) {
    fwrite($file2, "This is an appended message.\n");
    fclose($file2);
    echo "Append Successful<br>";
} else {
    echo "Error in appending file<br>";
}

// 🔹 3. COPYING FILE (example.txt → example_copy.txt)
if (file_exists("example.txt")) {
    if (copy("example.txt", "example_copy.txt")) {
        echo "Copy Successful<br>";
    } else {
        echo "Error in copying file<br>";
    }
} else {
    echo "example.txt not found<br>";
}

// 🔹 4. READING FROM FILE (example_copy.txt)
if (file_exists("example_copy.txt")) {

    $file3 = fopen("example_copy.txt", "r");

    if ($file3) {
        echo "<br>File Content:<br>";
        while (!feof($file3)) {
            echo fgets($file3) . "<br>";
        }
        fclose($file3);
    } else {
        echo "Error in reading file<br>";
    }

} else {
    echo "example_copy.txt not found<br>";
}

// 🔹 5. RENAMING FILE (sample.txt → renamed_sample.txt)
if (file_exists("sample.txt")) {
    if (rename("sample.txt", "renamed_sample.txt")) {
        echo "<br>Rename Successful<br>";
    } else {
        echo "Error in renaming file<br>";
    }
} else {
    echo "sample.txt not found<br>";
}

// 🔹 6. DELETING FILE (example_copy.txt)
if (file_exists("example_copy.txt")) {
    if (unlink("example_copy.txt")) {
        echo "Delete Successful<br>";
    } else {
        echo "Error in deleting file<br>";
    }
} else {
    echo "example_copy.txt not found<br>";
}

?>