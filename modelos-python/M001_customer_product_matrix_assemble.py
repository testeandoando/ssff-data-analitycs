
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
  
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
import pandas as pd
import numpy as np
from pyspark.sql.functions import current_date, current_timestamp,to_timestamp,to_date
df = pd.read_csv("s3://ssff-data-discovery/Data Analytics/pendiente_migracion/principalidad_rubro32_0822_1022.csv", encoding="latin-1" , low_memory=False)
df = df.set_index(keys="CTARUT")
cols_of_interest = [s for s in df.columns if "score" not in s.split("_")]
cols = [i.split("_")[1] for i in cols_of_interest]
df = df.iloc[:50000,:].copy()
df = df[cols_of_interest]
df.columns = cols
def str_to_scale(x):
    if x == "Outline":
        return 4
    elif x == "Frecuentes":
        return 3
    elif x == "Ocasionales":
        return 2
    elif x == "Puntuales":
        return 1
    else: 
        return 0
for col in cols:
    df[col] = df[col].apply(str_to_scale)
similarity_df = pd.DataFrame(columns=cols, index=cols)
N = len(cols)

for prod1 in cols:
    for prod2 in cols:
        similarity_df.loc[prod1, prod2] = np.dot(df[prod1], df[prod2])/(np.linalg.norm(df[prod1])*np.linalg.norm(df[prod2]))

similarity_df.fillna(0, inplace=True)
df_dummy = df.copy(deep=True)
df_test = df.copy(deep=True)
################### NEW CODE ###################
ruts = list(df_test.index)

for rut in ruts:
    for col in cols:
        if int(df_test.loc[rut, col]) == 0:
            pred = np.dot(np.array(similarity_df.loc[col, :]),np.array(df_dummy.loc[rut, :]))
            norm = np.sum(np.array(similarity_df.loc[col, :])) - 1.0
            df_test.loc[rut, col] = pred/norm
df_test.fillna(0, inplace=True)
#################################################
df.reset_index().rename(mapper={"index":"CTARUT"}, axis=1).to_csv("s3://ssff-data-analytics/resultados_modelos/data/PREDICTED_CUSTOMER_PRODUCT_MATRIX.csv", index=False)
spark_df = spark.createDataFrame(df_test.reset_index().rename(mapper={"index":"CTARUT"}, axis=1))
## Quitando simbolos invalidos del nombre de las columnas
spark_df1 = spark_df.withColumn("fecha", to_date(current_timestamp(),"yyy-MM-dd")) \
.withColumnRenamed("GRANDES TIENDAS", "GRANDES_TIENDAS") \
.withColumnRenamed("ALMACEN / MINIMARKET", "ALMACEN_MINIMARKET") \
.withColumnRenamed("SEGUROS Y SERVICIOS FINANCIEROS", "SEGUROS_Y_SERVICIOS_FINANCIEROST") \
.withColumnRenamed("EMPRESAS DE PAGO", "EMPRESAS_DE_PAGO") \
.withColumnRenamed("CENTROS MEDICOS", "CENTROS_MEDICOS") \
.withColumnRenamed("CARNICERIAS Y PESCADERIAS", "CARNICERIAS_Y_PESCADERIAS") \
.withColumnRenamed("BOTILLERIAS, VINOS Y LICORES", "BOTILLERIAS_VINOS_Y_LICORES") \
.withColumnRenamed("SERVICIOS AUTOMOTRICES", "SERVICIOS_AUTOMOTRICES") \
.withColumnRenamed("FERRETERIA, HOGAR Y CONSTRUCCION", "FERRETERIA_HOGAR_Y_CONSTRUCCION") \
.withColumnRenamed("CASA Y DECORACION", "CASA_Y_DECORACION") \
.withColumnRenamed("ESTACIONAMIENTO DE AUTOMOVILES", "ESTACIONAMIENTO_DE_AUTOMOVILES") \
.withColumnRenamed("ARTICULOS PARA CUMPLEANOS Y FIESTAS", "ARTICULOS_PARA_CUMPLEANOS_Y_FIESTAS") \
.withColumnRenamed("CONFITERIAS Y BOMBONERIAS", "CONFITERIAS_Y_BOMBONERIAS") \
.withColumnRenamed("ARTICULOS MEDICOS", "ARTICULOS_MEDICOS") \
.withColumnRenamed("MASCOTAS Y VETERINARIOS", "MASCOTAS_Y_VETERINARIOS") \
.withColumnRenamed("CARTERAS, MALETAS Y ACCESORIOS", "CARTERAS_MALETAS_Y_ACCESORIOS") \
.withColumnRenamed("ARTICULOS PARA BEBES Y NINOS", "ARTICULOS_PARA_BEBES_Y_NINOS") \
.withColumnRenamed("ARTICULOS PARA DEPORTE Y CAMPING", "ARTICULOS_PARA_DEPORTE_Y_CAMPING") \
.withColumnRenamed("PROFESIONALES RUBRO DE SALUD", "PROFESIONALES_RUBRO_DE_SALUD") \
.withColumnRenamed("AUTOPISTAS / PEAJES", "AUTOPISTAS_PEAJES") \
.withColumnRenamed("ARRIENDO DE AUTOMOVILES", "ARRIENDO_DE_AUTOMOVILES") \
.withColumnRenamed("SEGURO AUTOS", "SEGURO_AUTOS") \
.withColumnRenamed("SEGURO MASCOTAS", "SEGURO_MASCOTAS") \
.withColumnRenamed("SEGURO VIAJES", "SEGURO_VIAJES") \
spark_df1.write.partitionBy("fecha").option("compression","snappy").parquet("s3://ssff-data-analytics/resultados_modelos/Recomendador/customer-product_matrix_assemble/")
df_test = pd.read_csv("data/PREDICTED_CUSTOMER_PRODUCT_MATRIX.csv")
df_test
job.commit()