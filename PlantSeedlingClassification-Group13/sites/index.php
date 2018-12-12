<!DOCTYPE html>
<html>

<head>
<title>Plant Seedling Classifications</title>
</head>
<body>

<table border="0">

<h1> Plant seedlings classification</h1>

<p> Upload your pictures with seedlings and you will find out the species.</p>
<img src="seedlings.jpg" height="500"/></br></br></br>

<form action="index.php" method="POST" enctype="multipart/form-data">
	<input type="file" name="userfile"/></input>
	<input type="submit" value="Upload"/ ></input>
</form>
</table>
</body>
</html>

<?php  

	if(!empty($_FILES['userfile'])){
		$path="/Applications/XAMPP/xamppfiles/htdocs/sites/images/";
		$path=$path.basename($_FILES['userfile']['name']);

		$ext_error= false;
		$extentions= array('jpg','jpeg','gif','png');
		$file_ext=explode('.', $_FILES['userfile']['name']);
		$file_ext=end($file_ext);

		if (!in_array($file_ext, $extentions)){
			$ext_error=true;
		}

		if ($ext_error){
			echo "Invalid file!";
			exit(0);
		}
		else {
			echo "Successful uploading!";
		}

		move_uploaded_file($_FILES['userfile']['tmp_name'], $path);
		
	}
?>



