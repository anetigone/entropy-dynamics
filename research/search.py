"""Bounded public literature retrieval. Never writes credentials or request URLs."""
import os, json, time, pathlib, requests
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parent
QUERIES = [
    'Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance',
    'test time adaptation entropy collapse dynamics',
    'test time adaptation early stopping theory',
    'Layerwise Early Stopping for Test Time Adaptation',
    'On the Learnability of Test-Time Adaptation',
    'When is Test-Time Adaptation Identifiable From Unlabeled Evidence',
]

def main():
    log = {'date': '2026-10-24', 'openalex_key_present': bool(os.getenv('OPENALEX_API_KEY')), 'requests': []}
    for q in QUERIES:
        params = {'search': q, 'per-page': 8, 'filter': 'to_publication_date:2026-10-24',
                  'select': 'id,doi,display_name,publication_date,authorships,primary_location,open_access,cited_by_count,referenced_works'}
        if os.getenv('OPENALEX_API_KEY'):
            params['api_key'] = os.environ['OPENALEX_API_KEY']
        try:
            r = requests.get('https://api.openalex.org/works', params=params, timeout=25)
            entry = {'service': 'openalex', 'query': q, 'status': r.status_code}
            if r.ok:
                entry['results'] = r.json().get('results', [])
            log['requests'].append(entry)
        except requests.RequestException:
            log['requests'].append({'service': 'openalex', 'query': q, 'status': 'network_error'})
        time.sleep(.4)
    # One unauthenticated S2 request. Respect rate limit, no automatic retry.
    try:
        r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params={
            'query': QUERIES[0], 'limit': 3, 'fields': 'title,year,authors,url,externalIds,citationCount'}, timeout=25)
        log['requests'].append({'service': 'semantic_scholar', 'query': QUERIES[0], 'status': r.status_code,
                                'results': r.json().get('data', []) if r.ok else []})
    except requests.RequestException:
        log['requests'].append({'service': 'semantic_scholar', 'status': 'network_error'})
    (ROOT/'search_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
    ids = ['2606.26457v1','2605.28057v2','2402.09004v2','2604.17542v1','2609.11235v1','2311.18193v4']
    for aid in ids:
        try:
            r = requests.get('https://arxiv.org/html/'+aid, timeout=30)
            if r.ok:
                soup = BeautifulSoup(r.text, 'html.parser')
                (ROOT/(aid+'.txt')).write_text(soup.get_text(' ', strip=True), encoding='utf-8')
            print(aid, r.status_code)
        except requests.RequestException:
            print(aid, 'network_error')
    print('OpenAlex key available:', log['openalex_key_present'])
    print([(e['service'], e['status']) for e in log['requests']])

if __name__ == '__main__':
    main()
