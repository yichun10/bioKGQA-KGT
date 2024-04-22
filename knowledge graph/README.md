You can build your own knowledge graph locally by following these steps.

1、Download Neo4j Desktop.

Note: Requires Neo4j version 4.4.0 or higher.

2、Install APOC. It can be installed directly within the application.

3、Import the files entity.csv and relationships.csv into the application's import folder.

4、Execute the following code sequentially:

```bash
# Building entities
LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'Genesymbol'
CREATE (e:Genesymbol {id: id})
SET e += {`name`: line[3], `oncogene`: line[5]}
RETURN count(e) AS genesymbols_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'Cancer'
CREATE (e:Cancer {id: id})
SET e += {`name`: line[3], `description`: line[5], `name_en`: line[7]}
RETURN count(e) AS genesymbols_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'Drug'
CREATE (e:Drug{id: id})
SET e += {`name`: line[3], `name_en`: line[5], `description`: line[7], `class_type`: line[9], `nmpa_approved`: line[11], `fda_approved`: line[13]}
RETURN count(e) AS drug_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'GeneticDisease'
CREATE (e:GeneticDisease{id: id})
SET e += {`name`: line[3], `name_en`: line[5], `description`: line[7]}
RETURN count(e) AS drug_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'SnvFull'
CREATE (e:SnvFull{id: id})
SET e += {`name`: line[3], `biological_effect`: line[5], `oncogenic`: line[7], `variant_type`: line[9]}
RETURN count(e) AS drug_created

LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'ClinicalTrial'
CREATE (e:ClinicalTrial{id: id})
SET e += {`name`: line[3], `description`: line[5], `min_age`: line[7], `max_age`: line[9], `gender`: line[11]}
RETURN count(e) AS drug_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'CancerCell'
CREATE (e:CancerCell{id: id})
SET e += {`name`: line[3]}
RETURN count(e) AS drug_created

LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'Fusion'
CREATE (e:Fusion{id: id})
SET e += {`name`: line[3]}
RETURN count(e) AS drug_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'CancerAlias'
CREATE (e:CancerAlias{id: id})
SET e += {`name`: line[3]}
RETURN count(e) AS drug_created


LOAD CSV FROM "file:///entity.csv" AS line FIELDTERMINATOR ','
WITH line, line[0] AS id, line[1] AS label
WHERE label = 'DrugAlias'
CREATE (e:DrugAlias{id: id})
SET e += {`name`: line[3]}
RETURN count(e) AS drug_created
```

```bash
# Building relationships
LOAD CSV FROM "file:///relationships.csv" AS line FIELDTERMINATOR ',' 
MATCH (source{name: line[0]}), (target{name: line[2]})
CALL apoc.create.relationship(source, line[1], {}, target) YIELD rel 
RETURN count(rel)
```

Note: This is just a simple example of building a graph. You can customize the graph according to your own preferences.

