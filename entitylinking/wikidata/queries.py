import json
import urllib

from entitylinking.wikidata import wdaccess, wdscheme

sparql_get_entity_by_label = """
        {
              {GRAPH <http://wikidata.org/statements> { ?e2 e:P1549s/e:P1549v "%demonym"@en}} UNION
              {VALUES ?labelpredicate {rdfs:label skos:altLabel}
              GRAPH <http://wikidata.org/terms> {
                            ?e2 ?labelpredicate ?matchedlabel. ?matchedlabel bif:contains "%entitylabel"@en  }
                            FILTER ( lang(?matchedlabel) = "en" )
                            }
        }
        FILTER EXISTS {{ GRAPH g:statements {{ ?e2 ?p ?v }} }}
        FILTER ( NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q17442446}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q15474042}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q18616576}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q5707594}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q427626}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q16521}} }} &&
                 NOT EXISTS {{GRAPH g:instances {{?e2 rdf:type e:Q11173}} }}                               
            )
        BIND (STRLEN(?matchedlabel) as ?len)
        """

sparql_get_entity_classes = """
            {GRAPH <http://wikidata.org/instances> {?e2 rdf:type ?topic}}
            {GRAPH <http://wikidata.org/terms> {
                        ?topic rdfs:label  ?label. }
                        FILTER ( lang(?label) = "en" )
            }
        """


sparql_get_entity_labels = """
        {
        VALUES ?e2 { %entityids }
        VALUES ?labelpredicate {rdfs:label skos:altLabel}
        GRAPH <http://wikidata.org/terms> { ?e2 ?labelpredicate ?label }        
        }
        """

sparql_get_main_entity_label = """
        {
        GRAPH <http://wikidata.org/terms> { ?e2 rdfs:label ?label }
        FILTER ( lang(?label) = "en" )
        }
        """

sparql_get_year_from_entity = """
        {
        VALUES ?e2 { %entityids }
        GRAPH <http://wikidata.org/statements> { ?e2 base:time ?et. BIND (YEAR(?et) AS ?label) }
        }
        """

sparql_map_f_id = """
        {
          GRAPH <http://wikidata.org/statements> { ?e2 e:P646s/e:P646v "%otherkbid%" }
        }
        """


sparql_map_wikipedia_id = """
        {
            GRAPH <http://wikidata.org/sitelinks> { <%otherkbid%> schema:about ?e2 }
        }
        """

sparql_relation_any_direction = """
        {
        {GRAPH <http://wikidata.org/statements> { ?e1 ?rd ?m . ?m ?p ?e2 . }}
        UNION
        {GRAPH <http://wikidata.org/statements> { ?e2 ?p ?m . ?m ?rr ?e1 . }}
        }
        """


def query_get_entity_by_label(label_tokens, limit=wdaccess.GLOBAL_RESULT_LIMIT):
    """
    A method to look up a WikiData entity by a label.

    :param label_tokens: label tokens of the entity as a list
    :param limit: limit on the result list size
    :return: a query that can be executed against WikiData
    >>> len(wdaccess.query_wikidata(query_get_entity_by_label(["Obama"]))) > 0
    True
    >>> 'Q8027' in {r['e2'] for r in wdaccess.query_wikidata(query_get_entity_by_label(["Martin Luther King"]))}
    True
    >>> wdaccess.query_wikidata(query_get_entity_by_label(["thai"]))[0]['label']
    'Thailand'
    >>> 'Q221150' in {l['e2'] for l in wdaccess.query_wikidata(query_get_entity_by_label(["Vikings"]))}
    True
    >>> "Q846570" in {l['e2'] for l in wdaccess.query_wikidata(query_get_entity_by_label(["American"]))}
    True
    """
    label_tokens = list(label_tokens)
    query = wdaccess.sparql_inference_clause + wdaccess.sparql_prefix
    variables = []
    query += wdaccess.sparql_select
    query += "{"
    sparql_entity_label_inst = sparql_get_entity_by_label + sparql_get_main_entity_label
    sparql_entity_label_inst = sparql_entity_label_inst.replace("VALUES ?matchedlabel { %entitylabels }", "")
    sparql_entity_label_inst = sparql_entity_label_inst.replace("%linkbyname%", "")
    label_string = " ".join(label_tokens)
    label_string = label_string.replace("'", "")
    sparql_entity_label_inst = sparql_entity_label_inst.replace("%entitylabel", "'{}'".format(label_string))
    if len(label_tokens) == 1:
        sparql_entity_label_inst = sparql_entity_label_inst.replace("%demonym", "{}".format(label_string.title()))
    else:
        sparql_entity_label_inst = \
            sparql_entity_label_inst.replace("{GRAPH <http://wikidata.org/statements> { ?e2 e:P1549s/e:P1549v \"%demonym\"@en}} UNION", "")
    query += sparql_entity_label_inst
    query += "}"
    variables.append("?e2")
    variables.append("?matchedlabel")
    variables.append("?label")
    query = query.replace("%queryvariables%", " ".join(variables))
    if wdaccess.wdaccess_p['mode'] == "quality":
        query += wdaccess.sparql_close_order.format("?len")
    query += wdaccess.sparql_close.format(limit)
    return query


def query_get_entity_labels(entity, limit=10):
    """
    Construct a WikiData query to retrieve entity labels for the given entity id.

    :param entity: entity kbID
    :param limit: limit on the result list size
    :return: a WikiData query
    >>> wdaccess.query_wikidata(query_get_entity_labels("Q36"))  # doctest: +ELLIPSIS
    [{'label': 'Poland'}, {'label': 'Polen'}, {'label': 'Republic of Poland'}, ...]
    """
    query = sparql_get_entity_labels.replace("VALUES ?e2 { %entityids }", "")
    query = wdaccess.query_entity_base(query, entity, limit)
    query = query.replace("%queryvariables%", "?label")
    return query


def query_get_main_entity_label(entity):
    """
    Construct a WikiData query to retrieve the main entity label for the given entity id.

    :param entity: entity kbID
    :return: a WikiData query
    >>> wdaccess.query_wikidata(query_get_main_entity_label("Q36"), prefix=None)
    [{'label': 'Poland'}]
    """
    query = wdaccess.query_entity_base(sparql_get_main_entity_label, entity, 1)
    query = query.replace("skos:altLabel", "")
    query = query.replace("%queryvariables%", "?label")
    return query


def query_get_labels_for_entities(entities, limit_per_entity=10):
    """
    Construct a WikiData query to retrieve entity labels for the given list of entity ids.

    :param entities: entity kbIDs
    :param limit_per_entity: limit on the result list size (multiplied with the size of the entity list)
    :return: a WikiData query
    >>> wdaccess.query_wikidata(query_get_labels_for_entities(["Q36", "Q76"]))  # doctest: +ELLIPSIS
    [{'e2': 'Q36', 'label': 'Poland'}, ..., {'e2': 'Q76', 'label': 'Obama'}, ...]
    >>> wdaccess.query_wikidata(query_get_labels_for_entities(['VTfb0eeb812ca69194eaaa87efa0c6d51d']))
    [{'e2': 'VTfb0eeb812ca69194eaaa87efa0c6d51d', 'label': '1972'}]
    """
    if all(e[0] not in 'qQ' or '-' in e for e in entities):
        query = sparql_get_year_from_entity
    else:
        entities = [e for e in entities if '-' not in e and e[0] in 'pqPQ']
        query = sparql_get_entity_labels
    query = wdaccess.query_entity_base(query, limit=limit_per_entity * len(entities))
    query = query.replace("%entityids", " ".join(["e:" + entity for entity in entities]))
    query = query.replace("%queryvariables%", "?e2 ?label")
    return query


def query_get_entity_classes(entity, only_direct_type=False):
    """

    :param entity:
    :param only_direct_type:
    :return:
    >>> {'label': 'country', 'topic': 'http://www.wikidata.org/entity/Q6256'} in wdaccess.query_wikidata(query_get_entity_classes("Q36"), prefix=None)
    True
    """
    query = wdaccess.query_entity_base(sparql_get_entity_classes, entity, wdaccess.GLOBAL_RESULT_LIMIT)
    if not only_direct_type:
        query = wdaccess.sparql_inference_clause + query
    query = query.replace("%queryvariables%", "?label ?topic")
    return query


def _query_map_id(other_kb_id, sparql_query):
    query = wdaccess.sparql_prefix + wdaccess.sparql_select
    query += "{"
    sparql_label_entity_inst = sparql_query.replace("%otherkbid%", other_kb_id)
    query += sparql_label_entity_inst
    query += "}"
    query = query.replace("%queryvariables%", "?e2")
    query += wdaccess.sparql_close.format(1)
    return query


def query_map_freebase_id(f_id):
    """
    Map a Freebase id to a Wikidata entity

    :param f_id: Freebase id
    :return: a WikiData query
    >>> wdaccess.query_wikidata(query_map_freebase_id("/m/0d3k14"))
    [{'e2': 'Q9696'}]
    """
    return _query_map_id(f_id, sparql_map_f_id)


def query_map_wikipedia_id(wikipedia_article_id):
    """
    Map a Wikipedia id to a Wikidata entity

    :param wikipedia_article_id: Freebase id
    :return: a WikiData query
    >>> wdaccess.query_wikidata(query_map_wikipedia_id("John_F._Kennedy"))
    [{'e2': 'Q9696'}]
    """
    wikipedia_article_id = wikipedia_article_id.replace(wdscheme.WIKIPEDIA_PREFIX, "")
    wikipedia_article_id = urllib.parse.quote(wikipedia_article_id, safe="/:")
    wikipedia_article_id = wdscheme.WIKIPEDIA_PREFIX + wikipedia_article_id
    return _query_map_id(wikipedia_article_id, sparql_map_wikipedia_id)


def query_semantic_signature(entity):
    """
    Construct a WikiData query to retrieve the semantic singature (all related entities) of the given entity.

    :param entity: entity kbID
    :return: a WikiData query
    """
    query = wdaccess.sparql_prefix + wdaccess.sparql_select
    query += "{"
    sparql_sem_singature_inst = sparql_relation_any_direction
    sparql_sem_singature_inst = sparql_sem_singature_inst.replace("?e2", "e:" + entity)
    query += sparql_sem_singature_inst
    query += sparql_get_main_entity_label.replace("?e2", "?e1")
    query += "}"
    query = query.replace("%queryvariables%", "?label ?p ?e1")
    query += wdaccess.sparql_close.format(wdaccess.GLOBAL_RESULT_LIMIT)
    return query


def get_main_entity_label(entity):
    """
    Retrieve the main label of the given entity. None is returned if no label could be found.

    :param entity: entity KB ID
    :return: entity label as a string
    >>> get_main_entity_label("Q12143")
    'time zone'
    """
    results = wdaccess.query_wikidata(query_get_main_entity_label(entity))
    if results and 'label' in results[0]:
        return results[0]['label']
    return None


def map_f_id(f_id):
    """
    Map the given Freebase id to a Wikidata id

    :param f_id: Freebase id as a string
    :return: Wikidata kbID
    """
    f_id = f_id.replace(".", "/")
    if not f_id.startswith("/"):
        f_id = "/" + f_id
    results = wdaccess.query_wikidata(query_map_freebase_id(f_id))
    if results and 'e2' in results[0]:
        return results[0]['e2']
    return None


def map_wikipedia_id(wikipedia_article_id):
    """
    Map the given Wikipedia article URL (id) to a Wikidata id

    :param wikipedia_article_id: Wikipedia id as a string
    :return: Wikidata kbID
    >>> map_wikipedia_id("PIAS_Entertainment_Group")
    'Q7119302'
    >>> map_wikipedia_id("Swimming_(sport)")
    'Q31920'
    >>> map_wikipedia_id("José_Reyes_(shortstop)")
    'Q220096'
    >>> map_wikipedia_id("The_Twilight_Saga:_New_Moon")
    'Q116928'
    >>> map_wikipedia_id("betty_ford_center")
    'Q850360'
    >>> map_wikipedia_id("1976_democratic_national_convention")
    'Q16152917'
    """
    results = wdaccess.query_wikidata(query_map_wikipedia_id(wikipedia_article_id))
    if results and 'e2' in results[0]:
        return results[0]['e2']
    response = urllib.request.urlopen("https://en.wikipedia.org/w/api.php?action=query&redirects=1&format=json&prop=info&inprop=url&titles=" +
                                      urllib.parse.quote(wikipedia_article_id))
    encoding = response.info().get_content_charset("utf-8")
    json_response = json.loads(response.read().decode(encoding))
    if 'query' in json_response and 'pages' in json_response['query']:
        json_response = list(json_response['query']['pages'].items())
        k, value = json_response[0]
        if k != -1 and 'canonicalurl' in value:
            canonical_article_url = urllib.parse.unquote(value['canonicalurl'])
            results = wdaccess.query_wikidata(query_map_wikipedia_id(canonical_article_url))
            if results and 'e2' in results[0]:
                return results[0]['e2']

    capitalized = "_".join([token.title() for token in wikipedia_article_id.split("_")])
    if capitalized != wikipedia_article_id:
        return map_wikipedia_id(capitalized)
    return None


mapped_types_sorted = [
    'location', 'geographical object', 'watercourse', 'organization', 'political organization', 'event',
    'fictional character', 'social group', 'language', 'disease', 'sport', 'space object', 'astronomical object',
    'work of art','creative work','intellectual work', 'device', 'publication', 'software', 'human']

mapped_types = {
    'human': 'person',
    'location': 'location',
    'geographical object': 'location',
    'watercourse': 'location',
    'organization': 'organization',
    'political organization': 'organization',
    'event' : 'event',
    'fictional human': 'character',
    'fictional character': 'character',
    'social group': 'thing',
    'language': 'thing',
    'disease': 'thing',
    'sport': 'thing',
    'space object': 'thing',
    'astronomical object': 'thing',
    'work of art': "product",
    'creative work': "product",
    'intellectual work': "product",
    'device': "product",
    'publication': "product",
    'software': "product"}


def get_mapped_entity_type(entity_id):
    entity_types = wdaccess.query_wikidata(query_get_entity_classes(entity_id, only_direct_type=False), prefix=None)
    etype = None
    i = 0
    while etype is None and i < len(mapped_types_sorted):
        if mapped_types_sorted[i] in {t['label'] for t in entity_types}:
            etype = mapped_types_sorted[i]
            return mapped_types[etype]
        i += 1
    if etype is None:
        return "other"
    return etype


def get_semantic_signature(entity):
    """
    Extract the semantic signature (all related entities and relations) of the given entity.

    :param entity: Wikidata id as a string
    :return: list of strings of related entities and relations
    >>> len(get_semantic_signature("Q76")[0])
    518
    >>> get_semantic_signature("Q179641") is not None
    True
    >>> get_semantic_signature("Q1963799")
    (set(), set())
    >>> ('instance of', 'P31') in get_semantic_signature("Q15862")[1]
    True
    """
    results = wdaccess.query_wikidata(query_semantic_signature(entity))
    results = wdaccess.filter_relations(results, b='p', freq_threshold=10)
    related_entities = {(r['label'], r['e1'], r['p'][:-1]) for r in results}
    relations = {(wdscheme.property2label.get(r['p'][:-1], {}).get("label"), r['p'][:-1]) for r in results}
    return related_entities, relations


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())