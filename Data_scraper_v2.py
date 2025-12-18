"""
web_data_scraper.py
Enhanced language data scraper that collects real text from the web
"""

import os
import json
import time
import random
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    """Configuration for the web scraper"""
    max_sentences_per_language: int = 500
    max_pages_per_source: int = 3
    min_sentence_length: int = 20
    max_sentence_length: int = 200
    request_timeout: int = 10
    user_agents: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            ]

class WebDataScraper:
    def __init__(self, config: ScraperConfig = None):
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(self.config.user_agents)})
        
        # Language-specific sources (Wikipedia and news sites)
        self.language_sources = {
            'english': [
                'https://en.wikipedia.org/wiki/Main_Page',
                'https://www.bbc.com/news',
                'https://www.theguardian.com/international',
                'https://www.nytimes.com/',
                'https://www.nationalgeographic.com/',
                'https://www.scientificamerican.com/',
                'https://www.economist.com/',
            ],
            'german': [
                'https://de.wikipedia.org/wiki/Wikipedia:Hauptseite',
                'https://www.spiegel.de/',
                'https://www.zeit.de/index',
                'https://www.faz.net/aktuell/',
                'https://www.sueddeutsche.de/',
                'https://www.tagesschau.de/',
                'https://www.dw.com/de/themen/s-9077',
            ],
            'spanish': [
                'https://es.wikipedia.org/wiki/Wikipedia:Portada',
                'https://www.elmundo.es/',
                'https://www.elpais.com/',
                'https://www.abc.es/',
                'https://www.lavanguardia.com/',
                'https://www.20minutos.es/',
                'https://www.bbc.com/mundo',
            ],
            'french': [
                'https://fr.wikipedia.org/wiki/Wikip%C3%A9dia:Accueil_principal',
                'https://www.lemonde.fr/',
                'https://www.lefigaro.fr/',
                'https://www.liberation.fr/',
                'https://www.20minutes.fr/',
                'https://www.france24.com/fr/',
                'https://www.rtl.fr/',
            ],
            'italian': [
                'https://it.wikipedia.org/wiki/Pagina_principale',
                'https://www.corriere.it/',
                'https://www.repubblica.it/',
                'https://www.lastampa.it/',
                'https://www.ilsole24ore.com/',
                'https://www.ansa.it/',
                'https://www.rainews.it/',
            ]
        }
        
        # Language-specific text patterns for extraction
        self.language_patterns = {
            'english': {
                'paragraph_selectors': ['p', 'div.article-body', 'div.story-body', 'section.content'],
                'exclude_selectors': ['footer', 'nav', 'header', 'aside', 'script', 'style', '[class*="menu"]'],
            },
            'german': {
                'paragraph_selectors': ['p', 'div.article', 'div.text', 'section.content'],
                'exclude_selectors': ['footer', 'nav', 'header', 'aside', 'script', 'style', '[class*="menu"]'],
            },
            'spanish': {
                'paragraph_selectors': ['p', 'div.articulo', 'div.texto', 'section.contenido'],
                'exclude_selectors': ['footer', 'nav', 'header', 'aside', 'script', 'style', '[class*="menu"]'],
            },
            'french': {
                'paragraph_selectors': ['p', 'div.article', 'div.texte', 'section.contenu'],
                'exclude_selectors': ['footer', 'nav', 'header', 'aside', 'script', 'style', '[class*="menu"]'],
            },
            'italian': {
                'paragraph_selectors': ['p', 'div.articolo', 'div.testo', 'section.contenuto'],
                'exclude_selectors': ['footer', 'nav', 'header', 'aside', 'script', 'style', '[class*="menu"]'],
            }
        }
    
    def get_random_user_agent(self) -> str:
        """Return a random user agent"""
        return random.choice(self.config.user_agents)
    
    def fetch_url(self, url: str) -> str:
        """Fetch content from a URL"""
        try:
            self.session.headers.update({'User-Agent': self.get_random_user_agent()})
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""
    
    def extract_links(self, html: str, base_url: str, language: str) -> List[str]:
        """Extract relevant links from a page"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        domain = urlparse(base_url).netloc
        
        # Look for article links
        link_patterns = {
            'english': ['article', 'story', 'news', 'feature'],
            'german': ['artikel', 'nachricht', 'news', 'meldung'],
            'spanish': ['articulo', 'noticia', 'reportaje'],
            'french': ['article', 'nouvelle', 'actualite'],
            'italian': ['articolo', 'notizia', 'reportage']
        }
        
        patterns = link_patterns.get(language, [])
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text().lower()
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = f"https://{domain}{href}"
            elif not href.startswith('http'):
                continue
            
            # Check if link looks like an article
            is_article = any(pattern in href.lower() for pattern in patterns) or \
                        any(pattern in text for pattern in patterns)
            
            if is_article and domain in href:
                links.append(href)
        
        # Remove duplicates and limit
        return list(set(links))[:20]
    
    def extract_sentences(self, html: str, language: str) -> List[str]:
        """Extract clean sentences from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for selector in self.language_patterns[language]['exclude_selectors']:
            for element in soup.select(selector):
                element.decompose()
        
        # Extract text from relevant elements
        paragraphs = []
        for selector in self.language_patterns[language]['paragraph_selectors']:
            paragraphs.extend(soup.select(selector))
        
        # Extract and clean text
        sentences = []
        for p in paragraphs:
            text = p.get_text().strip()
            if text:
                # Split into sentences
                # Simple sentence splitting (improve with NLTK for production)
                text_sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in text_sentences:
                    clean_sentence = self.clean_sentence(sentence, language)
                    if clean_sentence:
                        sentences.append(clean_sentence)
        
        return sentences
    
    def clean_sentence(self, sentence: str, language: str) -> str:
        """Clean and validate a sentence"""
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Check length requirements
        if len(sentence) < self.config.min_sentence_length:
            return ""
        if len(sentence) > self.config.max_sentence_length:
            return ""
        
        # Language-specific validation
        min_words = 4
        if len(sentence.split()) < min_words:
            return ""
        
        # Remove sentences with too many numbers or special chars
        char_count = len(sentence)
        digit_count = sum(c.isdigit() for c in sentence)
        if digit_count > char_count * 0.3:  # More than 30% digits
            return ""
        
        # Remove sentences with URLs
        if 'http://' in sentence.lower() or 'https://' in sentence.lower():
            return ""
        
        # Remove sentences with email addresses
        if '@' in sentence and '.' in sentence.split('@')[-1]:
            return ""
        
        # Capitalize first letter
        if sentence and sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def scrape_language(self, language: str) -> List[str]:
        """Scrape data for a specific language"""
        logger.info(f"Starting scrape for {language}")
        sources = self.language_sources.get(language, [])
        all_sentences = set()
        total_scraped = 0
        
        for source_url in sources:
            if total_scraped >= self.config.max_sentences_per_language:
                break
            
            logger.info(f"Scraping from {source_url}")
            try:
                # Get main page
                main_html = self.fetch_url(source_url)
                if not main_html:
                    continue
                
                # Extract links from main page
                links = self.extract_links(main_html, source_url, language)
                pages_to_scrape = [source_url] + links[:self.config.max_pages_per_source]
                
                # Scrape each page
                for page_url in pages_to_scrape:
                    if total_scraped >= self.config.max_sentences_per_language:
                        break
                    
                    logger.info(f"  Scraping page: {page_url[:80]}...")
                    page_html = self.fetch_url(page_url)
                    if page_html:
                        sentences = self.extract_sentences(page_html, language)
                        
                        # Add unique sentences
                        new_sentences = [s for s in sentences if s not in all_sentences]
                        all_sentences.update(new_sentences)
                        total_scraped = len(all_sentences)
                        
                        logger.info(f"    Found {len(new_sentences)} new sentences (total: {total_scraped})")
                    
                    # Be polite - delay between requests
                    time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error scraping {source_url}: {e}")
                continue
        
        logger.info(f"Finished scraping {language}: {len(all_sentences)} sentences")
        return list(all_sentences)
    
    def scrape_all_languages(self) -> Dict[str, List[str]]:
        """Scrape data for all languages"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.scrape_language, lang): lang 
                for lang in self.language_sources.keys()
            }
            
            for future in as_completed(futures):
                language = futures[future]
                try:
                    sentences = future.result()
                    results[language] = sentences
                    logger.info(f"✓ {language}: {len(sentences)} sentences")
                except Exception as e:
                    logger.error(f"✗ Failed to scrape {language}: {e}")
                    results[language] = []
        
        return results
    
    def save_results(self, results: Dict[str, List[str]], output_dir: str = 'web_data'):
        """Save scraped results to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        stats = {}
        total_sentences = 0
        
        for language, sentences in results.items():
            filename = os.path.join(output_dir, f"{language}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
            
            stats[language] = len(sentences)
            total_sentences += len(sentences)
            logger.info(f"Saved {len(sentences)} sentences to {filename}")
        
        # Save statistics
        stats_file = os.path.join(output_dir, 'scraping_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_sentences': total_sentences,
                'languages': stats,
                'timestamp': time.time(),
                'date': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📊 Scraping Summary:")
        logger.info(f"   Total sentences: {total_sentences}")
        for lang, count in stats.items():
            logger.info(f"   {lang.capitalize():10}: {count:4} sentences")
        
        return stats

def augment_sentences(sentences: List[str]) -> List[str]:
    """Augment sentences with variations"""
    augmented = set(sentences)  # Start with original
    
    for sentence in sentences:
        # Skip if sentence is too short
        if len(sentence) < 10:
            continue
        
        # 1. Case variations
        augmented.add(sentence.lower())
        augmented.add(sentence.upper())
        
        # 2. Punctuation variations
        if sentence.endswith('.'):
            augmented.add(sentence[:-1] + '!')
            augmented.add(sentence[:-1] + '?')
        elif sentence.endswith('!'):
            augmented.add(sentence[:-1] + '.')
            augmented.add(sentence[:-1] + '?')
        elif sentence.endswith('?'):
            augmented.add(sentence[:-1] + '.')
            augmented.add(sentence[:-1] + '!')
        
        # 3. Add common prefixes
        prefixes = [
            "Note: ",
            "Important: ",
            "Remember: ",
            "FYI: ",
            "Tip: ",
        ]
        
        for prefix in prefixes:
            augmented.add(prefix + sentence)
        
        # 4. Remove or add spaces (simulate different formatting)
        words = sentence.split()
        if len(words) > 3:
            # Remove some spaces (simulate compound words)
            no_space_version = ''.join(words[:2]) + ' ' + ' '.join(words[2:])
            augmented.add(no_space_version)
            
            # Add extra spaces
            extra_space = '  '.join(words)
            augmented.add(extra_space)
        
        # 5. Simulate typos for common words
        typo_variations = {
            'the': 'teh',
            'and': 'adn',
            'that': 'taht',
            'with': 'withe',
            'this': 'thsi',
            'have': 'haev',
            'from': 'form',
        }
        
        for correct, typo in typo_variations.items():
            if f' {correct} ' in f' {sentence.lower()} ':
                typo_sentence = re.sub(
                    rf'\b{correct}\b', 
                    typo, 
                    sentence, 
                    flags=re.IGNORECASE
                )
                augmented.add(typo_sentence)
    
    return list(augmented)

def create_fallback_data() -> Dict[str, List[str]]:
    """Create fallback data in case web scraping fails"""
    logger.info("Creating fallback data...")
    
    # Expanded manual data for each language
    fallback_data = {
        'english': [
            # News & Current Affairs
            "The government announced new economic measures to address inflation concerns.",
            "Scientists have made a breakthrough in renewable energy technology this week.",
            "Global markets reacted positively to the latest trade agreement news.",
            "Healthcare reforms are being debated in parliament amid public consultations.",
            "Climate change conference delegates reached a consensus on emission targets.",
            
            # Technology
            "Artificial intelligence is transforming industries across the global economy.",
            "Cybersecurity experts warn about increasing threats to digital infrastructure.",
            "Quantum computing research shows promising results for complex calculations.",
            "Space exploration missions continue to expand our understanding of the universe.",
            "Blockchain technology is being adopted by financial institutions worldwide.",
            
            # Business & Finance
            "Corporate earnings reports exceeded analyst expectations this quarter.",
            "Startup companies are attracting significant venture capital investment.",
            "Economic indicators suggest steady growth in the manufacturing sector.",
            "International trade negotiations are progressing towards mutual agreements.",
            "Consumer confidence remains stable despite global economic uncertainties.",
            
            # Science & Health
            "Medical researchers discovered a new approach to treating chronic diseases.",
            "Environmental scientists published findings on ocean conservation efforts.",
            "Genetic studies reveal insights into human evolution and migration patterns.",
            "Public health initiatives successfully reduced infection rates this year.",
            "Nutrition experts emphasize the importance of balanced dietary habits.",
            
            # Education & Culture
            "Educational institutions are integrating digital learning platforms effectively.",
            "Cultural festivals celebrate diversity and promote community engagement.",
            "Literary awards recognize outstanding contributions to contemporary fiction.",
            "Historical preservation projects restore important architectural landmarks.",
            "Art exhibitions showcase emerging talent from various creative disciplines.",
        ],
        
        'german': [
            # Nachrichten & Aktuelles
            "Die Regierung kündigte neue Wirtschaftsmaßnahmen zur Bekämpfung der Inflation an.",
            "Wissenschaftler haben diese Woche einen Durchbruch in der Erneuerbare-Energien-Technologie erzielt.",
            "Die globalen Märkte reagierten positiv auf die neuesten Nachrichten zum Handelsabkommen.",
            "Gesundheitsreformen werden im Parlament debattiert, während öffentliche Konsultationen stattfinden.",
            "Delegierte der Klimawandelkonferenz erzielten einen Konsens über Emissionsziele.",
            
            # Technologie
            "Künstliche Intelligenz transformiert Industrien in der gesamten Weltwirtschaft.",
            "Cybersicherheitsexperten warnen vor zunehmenden Bedrohungen der digitalen Infrastruktur.",
            "Quantencomputing-Forschung zeigt vielversprechende Ergebnisse für komplexe Berechnungen.",
            "Weltraumforschungsmissionen erweitern weiterhin unser Verständnis des Universums.",
            "Blockchain-Technologie wird von Finanzinstituten weltweit übernommen.",
            
            # Wirtschaft & Finanzen
            "Unternehmensgewinnberichte übertrafen die Erwartungen der Analysten in diesem Quartal.",
            "Startup-Unternehmen ziehen erhebliche Venture-Capital-Investitionen an.",
            "Wirtschaftsindikatoren deuten auf ein stabiles Wachstum im verarbeitenden Gewerbe hin.",
            "Internationale Handelsverhandlungen schreiten in Richtung gegenseitiger Vereinbarungen voran.",
            "Das Verbrauchervertrauen bleibt trotz globaler wirtschaftlicher Unsicherheiten stabil.",
            
            # Wissenschaft & Gesundheit
            "Medizinforscher entdeckten einen neuen Ansatz zur Behandlung chronischer Krankheiten.",
            "Umweltwissenschaftler veröffentlichten Erkenntnisse zu Ozeanschutzbemühungen.",
            "Genetische Studien geben Einblicke in die menschliche Evolution und Migrationsmuster.",
            "Öffentliche Gesundheitsinitiativen senkten dieses Jahr erfolgreich die Infektionsraten.",
            "Ernährungsexperten betonen die Bedeutung ausgewogener Ernährungsgewohnheiten.",
            
            # Bildung & Kultur
            "Bildungseinrichtungen integrieren digitale Lernplattformen effektiv.",
            "Kulturfestivals feiern Vielfalt und fördern Engagement in der Gemeinschaft.",
            "Literarische Auszeichnungen würdigen herausragende Beiträge zur zeitgenössischen Literatur.",
            "Historische Denkmalschutzprojekte restaurieren wichtige architektonische Wahrzeichen.",
            "Kunstausstellungen präsentieren aufstrebende Talente aus verschiedenen kreativen Disziplinen.",
        ],
        
        'spanish': [
            # Noticias y Actualidad
            "El gobierno anunció nuevas medidas económicas para abordar las preocupaciones inflacionarias.",
            "Los científicos han logrado un avance en tecnología de energía renovable esta semana.",
            "Los mercados globales reaccionaron positivamente a las últimas noticias del acuerdo comercial.",
            "Las reformas sanitarias se debaten en el parlamento en medio de consultas públicas.",
            "Los delegados de la conferencia sobre cambio climático alcanzaron un consenso sobre los objetivos de emisiones.",
            
            # Tecnología
            "La inteligencia artificial está transformando industrias en toda la economía global.",
            "Los expertos en ciberseguridad advierten sobre amenazas crecientes a la infraestructura digital.",
            "La investigación en computación cuántica muestra resultados prometedores para cálculos complejos.",
            "Las misiones de exploración espacial continúan expandiendo nuestra comprensión del universo.",
            "La tecnología blockchain está siendo adoptada por instituciones financieras en todo el mundo.",
            
            # Negocios y Finanzas
            "Los informes de ganancias corporativas superaron las expectativas de los analistas este trimestre.",
            "Las empresas emergentes están atrayendo inversiones significativas de capital de riesgo.",
            "Los indicadores económicos sugieren un crecimiento constante en el sector manufacturero.",
            "Las negociaciones comerciales internacionales avanzan hacia acuerdos mutuos.",
            "La confianza del consumidor se mantiene estable a pesar de las incertidumbres económicas globales.",
            
            # Ciencia y Salud
            "Los investigadores médicos descubrieron un nuevo enfoque para tratar enfermedades crónicas.",
            "Los científicos ambientales publicaron hallazgos sobre los esfuerzos de conservación oceánica.",
            "Los estudios genéticos revelan información sobre la evolución humana y los patrones migratorios.",
            "Las iniciativas de salud pública redujeron exitosamente las tasas de infección este año.",
            "Los expertos en nutrición enfatizan la importancia de hábitos dietéticos equilibrados.",
            
            # Educación y Cultura
            "Las instituciones educativas están integrando plataformas de aprendizaje digital de manera efectiva.",
            "Los festivales culturales celebran la diversidad y promueven la participación comunitaria.",
            "Los premios literarios reconocen contribuciones sobresalientes a la ficción contemporánea.",
            "Los proyectos de preservación histórica restauran hitos arquitectónicos importantes.",
            "Las exposiciones de arte muestran talento emergente de varias disciplinas creativas.",
        ],
        
        'french': [
            # Actualités
            "Le gouvernement a annoncé de nouvelles mesures économiques pour répondre aux préoccupations inflationnistes.",
            "Les scientifiques ont réalisé une percée dans la technologie des énergies renouvelables cette semaine.",
            "Les marchés mondiaux ont réagi positivement aux dernières nouvelles de l'accord commercial.",
            "Les réformes de santé sont débattues au Parlement dans le cadre de consultations publiques.",
            "Les délégués de la conférence sur le changement climatique ont atteint un consensus sur les objectifs d'émissions.",
            
            # Technologie
            "L'intelligence artificielle transforme les industries à travers l'économie mondiale.",
            "Les experts en cybersécurité avertissent des menaces croissantes pour les infrastructures numériques.",
            "La recherche en informatique quantique montre des résultats prometteurs pour des calculs complexes.",
            "Les missions d'exploration spatiale continuent d'élargir notre compréhension de l'univers.",
            "La technologie blockchain est adoptée par les institutions financières du monde entier.",
            
            # Affaires et Finance
            "Les rapports sur les bénéfices des entreprises ont dépassé les attentes des analystes ce trimestre.",
            "Les startups attirent des investissements en capital-risque significatifs.",
            "Les indicateurs économiques suggèrent une croissance stable dans le secteur manufacturier.",
            "Les négociations commerciales internationales progressent vers des accords mutuels.",
            "La confiance des consommateurs reste stable malgré les incertitudes économiques mondiales.",
            
            # Science et Santé
            "Les chercheurs médicaux ont découvert une nouvelle approche pour traiter les maladies chroniques.",
            "Les scientifiques environnementaux ont publié des résultats sur les efforts de conservation des océans.",
            "Les études génétiques révèlent des informations sur l'évolution humaine et les schémas migratoires.",
            "Les initiatives de santé publique ont réduit avec succès les taux d'infection cette année.",
            "Les experts en nutrition soulignent l'importance d'habitudes alimentaires équilibrées.",
            
            # Éducation et Culture
            "Les établissements d'enseignement intègrent efficacement les plateformes d'apprentissage numérique.",
            "Les festivals culturels célèbrent la diversité et promeuvent l'engagement communautaire.",
            "Les prix littéraires reconnaissent les contributions exceptionnelles à la fiction contemporaine.",
            "Les projets de préservation historique restaurent des monuments architecturaux importants.",
            "Les expositions d'art présentent des talents émergents de diverses disciplines créatives.",
        ],
        
        'italian': [
            # Notizie e Attualità
            "Il governo ha annunciato nuove misure economiche per affrontare le preoccupazioni inflazionistiche.",
            "Gli scienziati hanno fatto una svolta nella tecnologia delle energie rinnovabili questa settimana.",
            "I mercati globali hanno reagito positivamente alle ultime notizie dell'accordo commerciale.",
            "Le riforme sanitarie vengono dibattute in parlamento durante le consultazioni pubbliche.",
            "I delegati della conferenza sul cambiamento climatico hanno raggiunto un consenso sugli obiettivi di emissione.",
            
            # Tecnologia
            "L'intelligenza artificiale sta trasformando le industrie in tutta l'economia globale.",
            "Gli esperti di cybersecurity avvertono delle minacce crescenti alle infrastrutture digitali.",
            "La ricerca sul calcolo quantistico mostra risultati promettenti per calcoli complessi.",
            "Le missioni di esplorazione spaziale continuano ad espandere la nostra comprensione dell'universo.",
            "La tecnologia blockchain viene adottata da istituzioni finanziarie in tutto il mondo.",
            
            # Affari e Finanza
            "I rapporti sugli utili aziendali hanno superato le aspettative degli analisti in questo trimestre.",
            "Le startup stanno attirando investimenti significativi di capitale di rischio.",
            "Gli indicatori economici suggeriscono una crescita costante nel settore manifatturiero.",
            "Le trattative commerciali internazionali stanno progredendo verso accordi reciproci.",
            "La fiducia dei consumatori rimane stabile nonostante le incertezze economiche globali.",
            
            # Scienza e Salute
            "I ricercatori medici hanno scoperto un nuovo approccio per trattare le malattie croniche.",
            "Gli scienziati ambientali hanno pubblicato risultati sugli sforzi di conservazione oceanica.",
            "Gli studi genetici rivelano informazioni sull'evoluzione umana e sui modelli migratori.",
            "Le iniziative di sanità pubblica hanno ridotto con successo i tassi di infezione quest'anno.",
            "Gli esperti di nutrizione sottolineano l'importanza di abitudini alimentari equilibrate.",
            
            # Educazione e Cultura
            "Le istituzioni educative stanno integrando efficacemente le piattaforme di apprendimento digitale.",
            "I festival culturali celebrano la diversità e promuovono il coinvolgimento della comunità.",
            "I premi letterari riconoscono contributi eccezionali alla narrativa contemporanea.",
            "I progetti di preservazione storica restaurano importanti punti di riferimento architettonici.",
            "Le mostre d'arte presentano talenti emergenti da varie discipline creative.",
        ]
    }
    
    return fallback_data

def main():
    """Main function"""
    print("=" * 70)
    print("ENHANCED WEB DATA SCRAPER")
    print("=" * 70)
    print("Scraping real text data from the web for 5 languages:")
    print("  • English")
    print("  • German")
    print("  • Spanish")
    print("  • French")
    print("  • Italian")
    print("\nNote: This will take several minutes to complete.")
    print("=" * 70)
    
    # Create output directory
    output_dir = 'web_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize scraper
        config = ScraperConfig(
            max_sentences_per_language=200,  # Reduced for faster testing
            max_pages_per_source=2,
            request_timeout=15
        )
        
        scraper = WebDataScraper(config)
        
        # Ask user for scraping method
        print("\nSelect scraping method:")
        print("1. Web scraping (requires internet, slower but real data)")
        print("2. Fallback data (faster, pre-prepared sentences)")
        print("3. Hybrid (try web scraping, fallback if fails)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        results = {}
        
        if choice == '1':
            print("\n🔄 Starting web scraping...")
            results = scraper.scrape_all_languages()
            
        elif choice == '2':
            print("\n📝 Loading fallback data...")
            results = create_fallback_data()
            
        elif choice == '3':
            print("\n🔄 Attempting web scraping first...")
            try:
                results = scraper.scrape_all_languages()
                # Check if we got enough data
                total_sentences = sum(len(v) for v in results.values())
                if total_sentences < 100:  # Not enough data
                    print(f"\n⚠️  Web scraping only collected {total_sentences} sentences.")
                    print("📝 Adding fallback data...")
                    fallback = create_fallback_data()
                    # Merge results, preferring web-scraped data
                    for lang in results:
                        if len(results[lang]) < 50:
                            results[lang].extend(fallback.get(lang, []))
                            results[lang] = list(set(results[lang]))[:100]
            except Exception as e:
                print(f"\n❌ Web scraping failed: {e}")
                print("📝 Falling back to pre-prepared data...")
                results = create_fallback_data()
        
        else:
            print("Invalid choice. Using hybrid approach.")
            results = create_fallback_data()
        
        # Augment the collected sentences
        print("\n🔧 Augmenting sentences...")
        augmented_results = {}
        total_augmented = 0
        
        for language, sentences in results.items():
            augmented = augment_sentences(sentences)
            augmented_results[language] = augmented
            total_augmented += len(augmented)
            print(f"  ✓ {language.capitalize():10}: {len(sentences):3} → {len(augmented):3} sentences")
        
        # Save results
        print(f"\n💾 Saving data...")
        stats = scraper.save_results(augmented_results, output_dir)
        
        # Create training-ready files
        print("\n📁 Creating training data files...")
        training_dir = 'data'
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        
        for language, sentences in augmented_results.items():
            filename = os.path.join(training_dir, f"{language}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
            print(f"  ✓ {language.capitalize():10}: {len(sentences)} sentences")
        
        print(f"\n✅ Data collection complete!")
        print(f"📊 Total sentences: {total_augmented}")
        print(f"📁 Training data saved to: {os.path.abspath(training_dir)}")
        print(f"📁 Raw web data saved to: {os.path.abspath(output_dir)}")
        print("\n🎯 Next: Run 'python enhanced_language_detector.py'")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user.")
        print("Creating fallback data instead...")
        
        # Create fallback data
        results = create_fallback_data()
        training_dir = 'data'
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        
        for language, sentences in results.items():
            filename = os.path.join(training_dir, f"{language}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence + '\n')
        
        print(f"✅ Fallback data created with {sum(len(v) for v in results.values())} sentences")
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()