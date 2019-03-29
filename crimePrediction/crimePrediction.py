import csv
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import FloatType,StringType,IntegerType,StructType
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,col
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL crime prediction functions") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def getNullValues(data):
    spark = init_spark()
    rowsData = ()
    rowDataPorc=()
    rowDataNames = ()
    for i in data.columns:
        rowDataNames = rowDataNames + (i + "_null",)
        aux=data.where(col(i).isNull()).count()
        porAux=(aux/data.count())*100
        rowsData = rowsData + (float(aux),)
        rowDataPorc=rowDataPorc+(float(porAux),)
    nullCount = spark.createDataFrame([rowsData,rowDataPorc], rowDataNames)
    return nullCount

def datasetWellbeing(demographicsFilename,economicsFilename,educationFilename, variable):

    #   result=spark.read.option("header","false").option("inferSchema","true").schema(StructType([StructField("plant",StringType())])).text(filename)
    #   result = result.select("*").limit(n).withColumn("id", monotonically_increasing_id())
    #   eliminateFisrt = udf(lambda x: x[1:],ArrayType(StringType()))
    #aggrSep=udf(lambda x:addSep(x),ArrayType(StringType()))
    #   result=result.select('id', split('plant',',')[0].alias('plant'),eliminateFisrt(split('plant',',')).alias('items'))
    #result=result.select('id','plant',aggrSep('items').alias('items'))

    spark = init_spark()

    #Selecting features from the dataframe

    eliminateSpace = udf(lambda x: x.replace(" ","-").lower(),StringType())
    porcentaj=udf(lambda x: x/100,FloatType())
    dataWellbeingDemographics = spark.read.csv(demographicsFilename, header=True, mode="DROPMALFORMED",inferSchema=True)
    dataWellbeingDemographics=dataWellbeingDemographics.select(eliminateSpace('Neighbourhood').alias('neighbourhood_d'),dataWellbeingDemographics['Neighbourhood Id'].alias("neighbourhood_id_d"),dataWellbeingDemographics['Total Area'].alias('total_area'),dataWellbeingDemographics['Total Population'].alias('total_population'))

    dataWellbeingEconomics = spark.read.csv(economicsFilename, header=True, mode="DROPMALFORMED",inferSchema=True)
    dataWellbeingEconomics = dataWellbeingEconomics.select(eliminateSpace('Neighbourhood').alias('neighbourhood_e'),
                                                                 dataWellbeingEconomics['Neighbourhood Id'].alias(
                                                                     "neighbourhood_id_e"),
                                                                 dataWellbeingEconomics['Home Prices'].alias(
                                                                     'home_prices'),
                                                                 dataWellbeingEconomics['Local Employment'].alias(
                                                                     'local_employment'),dataWellbeingEconomics['Social Assistance Recipients'].alias('social_assistance_recipients'))

    dataWellbeingEducation = spark.read.csv(educationFilename, header=True, mode="DROPMALFORMED",inferSchema=True)
    dataWellbeingEducation = dataWellbeingEducation.select(eliminateSpace('Neighbourhood').alias('neighbourhood_ed'),
                                                           dataWellbeingEducation['Neighbourhood Id'].alias(
                                                               "neighbourhood_id_ed"),
                                                           dataWellbeingEducation['Catholic School Graduation'].alias(
                                                               'catholic_school_graduation'),
                                                           porcentaj(dataWellbeingEducation['Catholic School Literacy']).alias(
                                                               'catholic_school_literacy'),
                                                           porcentaj(dataWellbeingEducation['Catholic University Applicants']).alias(
                                                               'catholic_university_applicants'))

    data = dataWellbeingEducation.join(dataWellbeingEconomics,dataWellbeingEducation['neighbourhood_id_ed']==dataWellbeingEconomics['neighbourhood_id_e'])
    data = data.join(dataWellbeingDemographics,
                                       data['neighbourhood_id_ed'] == dataWellbeingDemographics[
                                           'neighbourhood_id_d'])

    data=data.select(data['neighbourhood_e'].alias('neighbourhood_w'),data['neighbourhood_id_e'].alias('neighbourhood_id'),data['total_area'],data['total_population'],data['home_prices'],data['local_employment'],data['social_assistance_recipients'],data['catholic_school_graduation'],data['catholic_school_literacy'],data['catholic_university_applicants'])

    if(variable==1):
        # Show Schema
        data.printSchema()

        #Counting null values
        nullCount=getNullValues(data)
        nullCount.show()

        #Feature metrics
        data.describe().show()

    return data


def dataSetCreactionCrime(crimeFilename,variable):
    spark = init_spark()
    dataCrimes = spark.read.csv(crimeFilename, header=True, mode="DROPMALFORMED",inferSchema=True)

    # Counting null values
    nullCount = getNullValues(dataCrimes)

    #Eliminate null values
    numberNotNull=dataCrimes.where(dataCrimes["occurrenceyear"].isNotNull() & dataCrimes["occurrencemonth"].isNotNull() & dataCrimes["occurrenceday"].isNotNull() & dataCrimes["occurrencedayofyear"].isNotNull() & dataCrimes["occurrencedayofweek"].isNotNull())

    if(variable==1):
        #Show Schema
        dataCrimes.printSchema()

        #Showing null
        nullCount.show()

        # Feature metrics
        dataCrimes.describe().show()


    return numberNotNull


def joinDataset(variable):

    rddCrime=dataSetCreactionCrime("./data/MCI_2014_to_2017.csv",0)
    rddWellbeing=datasetWellbeing("./data/WB-Demographics.csv","./data/WB-Economics.csv","./data/WB-Education.csv",0)

    data = rddCrime.join(rddWellbeing,rddCrime['Hood_ID'] == rddWellbeing['neighbourhood_id'])
    data = data.select(data['neighbourhood_w'].alias("neighbourhood_name"),
                       data['neighbourhood_id'], data['total_area'],
                       data['total_population'], data['home_prices'], data['local_employment'],
                       data['social_assistance_recipients'], data['catholic_school_graduation'],
                       data['catholic_school_literacy'], data['catholic_university_applicants'],
                       data['neighbourhood'],data['X'].alias("x"),data['Y'].alias("y"),data['Index_'].alias("index"),
                       data["event_unique_id"], data["premisetype"],data["ucr_code"],data["ucr_ext"],data["offence"],
                       data["reportedyear"],data["reportedmonth"], data["reportedday"],data["reporteddayofyear"],data["reporteddayofweek"],
                       data["reportedhour"], data["occurrenceyear"],data["occurrencemonth"],data["occurrenceday"],data["occurrencedayofyear"],
                       data["occurrencedayofweek"], data["occurrencehour"], data["MCI"].alias("mci"),data["Lat"].alias("lat"),data["Long"].alias("long"), data["FID"].alias("fid"))

    if(variable==1):
        data.printSchema()
    return data

def groupingAl():

    result=joinDataset(0)
    result.printSchema()
    result.groupBy("mci").count().orderBy(col("count").desc()).show()
    result.groupBy("neighbourhood_name").count().orderBy(col("count").desc()).show()
    analisis=result.withColumn("year_difference",col("reportedyear")-col("occurrenceyear"))
    analisis.printSchema()
    analisis.select(analisis["year_difference"],analisis["index"]).where(col("year_difference")>5).orderBy(col("year_difference").desc()).show()
    return result

groupingAl()