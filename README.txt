# Ejecutar especificando nombre y programa scala
# Run scripts through Spark-shell, specify task name and script name
	/opt/spark-1.6.2/bin/spark-shell --name <task_name> -i <file>.scala
# Ejecutar especificando nombre y programa scala con dependencias de spark-packages
# Run scripts through Spark-shell, specify task name and script name. Also use spark packages
	/opt/spark-1.6.2/bin/spark-shell --packages sramirez:spark-infotheoretic-feature-selection:1.3.1,sramirez:spark-MDLP-discretization:1.2.1 --name <task_name> -i <file>.scala
# Ejecutar especificando nombre y programa scala con dependencias de spark-packages, limitando recursos
# Run scripts through Spark-shell, specify task name and script name, using spark packages. Limit RAM and CPU use
	/opt/spark-1.6.2/bin/spark-shell --packages sramirez:spark-infotheoretic-feature-selection:1.3.1,sramirez:spark-MDLP-discretization:1.2.1 --name <task_name> -i <file>.scala --executor-memory 16G --total-executor-cores 20
# Ejecutar especificando nombre y programa scala con dependencias de spark-packages y .jar, limitando recursos
# Run scripts through Spark-shell, specify task name and script name, using spark packages, and implementation in Jar file. Limit RAM and CPU use
	/opt/spark-1.6.2/bin/spark-shell --packages sramirez:spark-infotheoretic-feature-selection:1.3.1,sramirez:spark-MDLP-discretization:1.2.1 --name <task_name> -i <file>.scala --jars Imb-sampling-1.2.jar --executor-memory 16G --total-executor-cores 32