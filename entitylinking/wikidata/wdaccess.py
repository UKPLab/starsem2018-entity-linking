import atexit
import json
import logging
import os

from SPARQLWrapper import SPARQLWrapper, JSON

from entitylinking.wikidata import wdscheme

wdaccess_p = {
    'backend': "http://knowledgebase:8890/sparql",
    'timeout': 20,
    'global_result_limit': 1000,
    'logger': logging.getLogger(__name__),
    'use.cache': False,
    'mode': "quality"  # options: precision, fast
}

logger = wdaccess_p['logger']
logger.setLevel(logging.INFO)


def set_backend(backend_url):
    global sparql
    sparql = SPARQLWrapper(backend_url)
    sparql.setReturnFormat(JSON)
    sparql.setMethod("GET")
    sparql.setTimeout(wdaccess_p.get('timeout', 40))


sparql = None
set_backend(wdaccess_p.get('backend', "http://knowledgebase:8890/sparql"))
GLOBAL_RESULT_LIMIT = wdaccess_p['global_result_limit']

sparql_inference_clause = """
        DEFINE input:inference 'instances'
        """

sparql_transitive_option = "option (transitive,t_no_cycles, t_min (1), t_max(5))"

sparql_prefix = """
        PREFIX g:<http://wikidata.org/>
        PREFIX e:<http://www.wikidata.org/entity/>
        PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
        PREFIX base:<http://www.wikidata.org/ontology#>
        PREFIX schema:<http://schema.org/>
        """

sparql_select = """
        SELECT DISTINCT %queryvariables% WHERE
        """

sparql_ask = """
        ASK WHERE
        """


sparql_close_order = " ORDER BY {}"
sparql_close = " LIMIT {}"

FILTER_RELATION_CLASSES = "qr"

query_cache = {}
cached_counter = 0
query_counter = 1

cache_location = os.path.abspath(__file__)
cache_location = os.path.dirname(cache_location)
if wdaccess_p['use.cache'] and os.path.isfile(cache_location + "/.wdacess.cache"):
    try:
        with open(cache_location + "/.wdacess.cache") as f:
            query_cache = json.load(f)
        logger.info("Query cache loaded. Size: {}".format(len(query_cache)))
    except Exception as ex:
        logger.error("Query cache exists, but can't be loaded. {}".format(ex))


def clear_cache():
    global query_cache
    query_cache = {}


def dump_cache():
    if wdaccess_p['use.cache']:
        logger.info("Cached query ratio: {}.".format(cached_counter / query_counter))
    if query_cache:
        logger.info("Dump query cache.")
        with open(cache_location + "/.wdacess.cache", "w") as out:
            json.dump(query_cache, out)


atexit.register(dump_cache)


def filter_relations(results, b='p', freq_threshold=0):
    """
    Takes results of a SPARQL query and filters out all rows that contain blacklisted relations.

    :param results: results of a SPARQL query
    :param b: the key of the relation value in the results dictionary
    :return: filtered results
    >>> filter_relations([{"p":"http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "e2":"http://www.wikidata.org/ontology#Item"}, {"p":"http://www.wikidata.org/entity/P1429s", "e2":"http://www.wikidata.org/entity/Q76S69dc8e7d-4666-633e-0631-05ad295c891b"}])
    []
    """
    results = [r for r in results if b not in r or
               (r[b][:-1] in wdscheme.content_properties and r[b][-1] not in FILTER_RELATION_CLASSES)
               ]
    results = [r for r in results if b not in r or wdscheme.property2label.get(r[b][:-1], {}).get('freq') > freq_threshold]
    return results


def query_wikidata(query, prefix=wdscheme.WIKIDATA_ENTITY_PREFIX, use_cache=-1, timeout=-1):
    """
    Execute the following query against WikiData
    :param query: SPARQL query to execute
    :param prefix: if supplied, then each returned URI should have the given prefix. The prefix is stripped
    :param use_cache: set to 0 or 1 to override the global setting
    :param timeout: set to a value large than 0 to override the global setting
    :return: a list of dictionaries that represent the queried bindings
    """
    use_cache = (wdaccess_p['use.cache'] and use_cache != 0) or use_cache == 1
    global query_counter, cached_counter, query_cache
    query_counter += 1
    if use_cache and query in query_cache:
        cached_counter += 1
        return query_cache[query]
    if timeout > 0:
        sparql.setTimeout(timeout)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except Exception as inst:
        logger.debug(inst)
        return []
    # Change the timeout back to the default
    if timeout > 0:
        sparql.setTimeout(wdaccess_p.get('timeout', 40))
    if "results" in results and len(results["results"]["bindings"]) > 0:
        results = results["results"]["bindings"]
        logger.debug("Results bindings: {}".format(results[0].keys()))
        if prefix:
            results = [r for r in results if all(not r[b]['value'].startswith("http://") or r[b]['value'].startswith(prefix) for b in r)]
        results = [{b: (r[b]['value'].replace(prefix, "") if prefix else r[b]['value']) for b in r} for r in results]
        if use_cache:
            query_cache[query] = results
        return results
    elif "boolean" in results:
        return results['boolean']
    else:
        logger.debug(results)
        return []


def query_base(limit=None):
    query = sparql_prefix + sparql_select
    query += "{"
    query += "%mainquery%"
    query += "}"
    if limit:
        query += sparql_close.format(limit)
    return query, "%mainquery%"


def query_entity_base(sparql_template, entity=None, limit=None):
    query, mainquery_placeholder = query_base(limit)
    if entity:
        sparql_template = sparql_template.replace("?e2", "e:" + entity)
    query = query.replace(mainquery_placeholder, sparql_template)
    return query


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
