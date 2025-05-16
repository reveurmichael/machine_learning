"""
Quotes.toscrape.com Scraper

This script demonstrates how to use the llm_selenium_agent package
to scrape quotes, authors, and tags from quotes.toscrape.com.
It can navigate through multiple pages and save the data to a CSV file.
"""

from llm_selenium_agent import BaseSeleniumChrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import csv
import time
import pandas as pd
import os
import json

class QuoteScraper(BaseSeleniumChrome):
    """A class to scrape quotes from quotes.toscrape.com."""
    
    def __init__(self):
        """Initialize the scraper with the target URL."""
        super().__init__()
        self.url = "https://quotes.toscrape.com/"
        self.quotes_data = []
        self.authors_data = {}  # Store author details keyed by name
    
    def login(self):
        """No login required for this site, but method is required by BaseSelenium."""
        pass
    
    def verify_login_success(self):
        """Method required by BaseSelenium."""
        return True
    
    def extract_quotes_from_page(self):
        """Extract all quotes from the current page."""
        # Wait for quotes to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "quote"))
        )
        
        # Find all quote elements
        quote_elements = self.driver.find_elements(By.CLASS_NAME, "quote")
        page_quotes = []
        
        # Process each quote
        for quote_element in quote_elements:
            # Extract text
            text = quote_element.find_element(By.CLASS_NAME, "text").text
            # Remove quotation marks from the beginning and end
            text = text[1:-1] if text.startswith('"') and text.endswith('"') else text
            
            # Extract author
            author = quote_element.find_element(By.CLASS_NAME, "author").text
            
            # Extract author URL
            author_url = quote_element.find_element(By.CSS_SELECTOR, "a[href*='author']").get_attribute("href")
            
            # Extract tags
            tags = [tag.text for tag in quote_element.find_elements(By.CLASS_NAME, "tag")]
            
            # Create quote data dictionary
            quote_data = {
                'text': text,
                'author': author,
                'author_url': author_url,
                'tags': tags
            }
            
            # Add to our page quotes list
            page_quotes.append(quote_data)
            
        # Add to our full data list
        self.quotes_data.extend(page_quotes)
        print(f"Extracted {len(page_quotes)} quotes from the current page.")
        return page_quotes
    
    def has_next_page(self):
        """Check if there is a 'Next' page button."""
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, "li.next > a")
            return True
        except NoSuchElementException:
            return False
    
    def go_to_next_page(self):
        """Navigate to the next page."""
        next_button = self.driver.find_element(By.CSS_SELECTOR, "li.next > a")
        next_button.click()
        
        # Wait for the page to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "quote"))
        )
        time.sleep(1)  # Add a small delay to ensure page is fully loaded
    
    def scrape_all_pages(self, max_pages=None):
        """
        Scrape quotes from all pages or up to max_pages.
        
        Args:
            max_pages: Maximum number of pages to scrape. None for all pages.
        """
        page_num = 1
        
        self.navigate_to_url()
        
        while True:
            print(f"Scraping page {page_num}...")
            self.extract_quotes_from_page()
            
            # Take a screenshot of the current page
            self.capture_screenshot()
            
            # Check if we reached the maximum pages
            if max_pages and page_num >= max_pages:
                print(f"Reached maximum pages ({max_pages}). Stopping.")
                break
            
            # Check if there's a next page
            if self.has_next_page():
                self.go_to_next_page()
                page_num += 1
            else:
                print("No more pages to scrape.")
                break
        
        print(f"Scraping completed. Total quotes scraped: {len(self.quotes_data)}")
    
    def get_author_info(self, author_url):
        """
        Navigate to an author's page and extract their information.
        
        Args:
            author_url: URL to the author's page
        
        Returns:
            Dict containing author's information
        """
        # Save the current URL to come back later
        current_url = self.driver.current_url
        
        # Navigate to the author's page
        self.driver.get(author_url)
        
        # Wait for the author information to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "author-details"))
        )
        
        # Extract information
        author_name = self.driver.find_element(By.CLASS_NAME, "author-title").text
        born_date = self.driver.find_element(By.CLASS_NAME, "author-born-date").text
        born_location = self.driver.find_element(By.CLASS_NAME, "author-born-location").text
        description = self.driver.find_element(By.CLASS_NAME, "author-description").text
        
        # Take a screenshot
        self.capture_screenshot()
        
        # Navigate back to the original page
        self.driver.get(current_url)
        
        # Wait for the page to load again
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "quote"))
        )
        
        return {
            'name': author_name,
            'born_date': born_date,
            'born_location': born_location,
            'description': description
        }
    
    def collect_all_author_info(self):
        """Collect detailed information for all authors."""
        # Get unique authors and their URLs
        unique_authors = {}
        for quote in self.quotes_data:
            if quote['author'] not in unique_authors:
                unique_authors[quote['author']] = quote['author_url']
        
        # Collect details for each author
        print(f"Collecting information for {len(unique_authors)} authors...")
        for author_name, author_url in unique_authors.items():
            print(f"Getting info for {author_name}...")
            self.authors_data[author_name] = self.get_author_info(author_url)
        
        print(f"Collected information for {len(self.authors_data)} authors.")
    
    def login_to_site(self, username, password):
        """
        Login to the quotes.toscrape.com website.
        
        Args:
            username: Username to use
            password: Password to use
        
        Returns:
            Boolean indicating success or failure
        """
        # Navigate to the login page
        self.driver.get("https://quotes.toscrape.com/login")
        
        # Wait for the login form to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        
        # Enter username and password
        username_field = self.driver.find_element(By.ID, "username")
        password_field = self.driver.find_element(By.ID, "password")
        
        username_field.send_keys(username)
        password_field.send_keys(password)
        
        # Submit the form
        password_field.submit()
        
        # Check if login was successful (look for the logout link)
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.LINK_TEXT, "Logout"))
            )
            return True
        except:
            return False
    
    def filter_quotes_by_author(self, author_name):
        """Filter and return quotes by a specific author."""
        return [quote for quote in self.quotes_data if quote['author'].lower() == author_name.lower()]
    
    def filter_quotes_by_tag(self, tag_name):
        """Filter and return quotes with a specific tag."""
        return [quote for quote in self.quotes_data if tag_name.lower() in [t.lower() for t in quote['tags']]]
    
    def save_to_csv(self, filename="quotes.csv"):
        """Save the scraped quotes to a CSV file."""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            # Create a CSV writer
            fieldnames = ['text', 'author', 'author_url', 'tags']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write the header
            writer.writeheader()
            
            # Write the data
            for quote in self.quotes_data:
                # Convert tags list to a string
                quote_to_write = quote.copy()
                quote_to_write['tags'] = ', '.join(quote_to_write['tags'])
                writer.writerow(quote_to_write)
        
        print(f"Data saved to {filename}")
    
    def save_authors_to_csv(self, filename="authors.csv"):
        """Save the author information to a CSV file."""
        if not self.authors_data:
            print("No author data to save.")
            return
            
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            # Create a CSV writer
            fieldnames = ['name', 'born_date', 'born_location', 'description']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write the header
            writer.writeheader()
            
            # Write the data
            for author_info in self.authors_data.values():
                writer.writerow(author_info)
        
        print(f"Author data saved to {filename}")
    
    def save_to_json(self, quotes_filename="quotes.json", authors_filename="authors.json"):
        """Save the scraped data to JSON files."""
        # Save quotes
        os.makedirs(os.path.dirname(quotes_filename) if os.path.dirname(quotes_filename) else '.', exist_ok=True)
        with open(quotes_filename, 'w', encoding='utf-8') as file:
            json.dump(self.quotes_data, file, ensure_ascii=False, indent=4)
        print(f"Quotes saved to {quotes_filename}")
        
        # Save authors if available
        if self.authors_data:
            os.makedirs(os.path.dirname(authors_filename) if os.path.dirname(authors_filename) else '.', exist_ok=True)
            with open(authors_filename, 'w', encoding='utf-8') as file:
                json.dump(self.authors_data, file, ensure_ascii=False, indent=4)
            print(f"Author data saved to {authors_filename}")
    
    def generate_pandas_dataframe(self):
        """Generate a pandas DataFrame from the quotes data."""
        # Convert the data to a format suitable for a DataFrame
        quotes_for_df = []
        for quote in self.quotes_data:
            quote_copy = quote.copy()
            quote_copy['tags'] = ', '.join(quote_copy['tags'])
            quotes_for_df.append(quote_copy)
        
        # Create the DataFrame
        df = pd.DataFrame(quotes_for_df)
        return df
    
    def analyze_data(self):
        """Perform basic analysis on the scraped data."""
        if not self.quotes_data:
            print("No data to analyze.")
            return
        
        # Create a DataFrame for easier analysis
        df = self.generate_pandas_dataframe()
        
        # Count quotes by author
        author_counts = df['author'].value_counts()
        print("\n=== Quotes by Author ===")
        for author, count in author_counts.items():
            print(f"{author}: {count} quotes")
        
        # Analyze tags
        all_tags = []
        for tags_str in df['tags']:
            all_tags.extend([tag.strip() for tag in tags_str.split(',')])
        
        tag_counts = pd.Series(all_tags).value_counts()
        print("\n=== Most Common Tags ===")
        for tag, count in tag_counts.head(10).items():
            print(f"{tag}: {count} occurrences")
        
        # Quote length analysis
        df['quote_length'] = df['text'].apply(len)
        avg_length = df['quote_length'].mean()
        min_length = df['quote_length'].min()
        max_length = df['quote_length'].max()
        
        print("\n=== Quote Length Statistics ===")
        print(f"Average quote length: {avg_length:.2f} characters")
        print(f"Shortest quote: {min_length} characters")
        print(f"Longest quote: {max_length} characters")
        
        # Find shortest and longest quotes
        shortest_quote = df.loc[df['quote_length'].idxmin()]
        longest_quote = df.loc[df['quote_length'].idxmax()]
        
        print("\nShortest quote:")
        print(f'"{shortest_quote["text"]}" by {shortest_quote["author"]}')
        
        print("\nLongest quote:")
        print(f'"{longest_quote["text"]}" by {longest_quote["author"]}')


def main():
    """Run the quote scraper program."""
    print("=== Quotes to Scrape Crawler ===")
    scraper = QuoteScraper()
    
    # We'll use a try-finally block to ensure the webdriver is terminated
    try:
        print("Setting up the environment...")
        scraper.prepare_environment()
        
        # Scrape quotes from all pages (or limit to a specific number)
        print("Starting to scrape quotes...")
        scraper.scrape_all_pages(max_pages=2)  # Limit to 2 pages for this example
        
        # Also get detailed author information
        print("Collecting author information...")
        scraper.collect_all_author_info()
        
        # Save the scraped data
        data_dir = "scraped_data"
        os.makedirs(data_dir, exist_ok=True)
        
        print("Saving data to files...")
        scraper.save_to_csv(f"{data_dir}/quotes.csv")
        scraper.save_authors_to_csv(f"{data_dir}/authors.csv")
        scraper.save_to_json(f"{data_dir}/quotes.json", f"{data_dir}/authors.json")
        
        # Perform some basic analysis
        print("Analyzing the scraped data...")
        scraper.analyze_data()
        
        # Example of filtering quotes
        print("\n=== Filtered Quotes ===")
        einstein_quotes = scraper.filter_quotes_by_author("Albert Einstein")
        print(f"Found {len(einstein_quotes)} quotes by Albert Einstein.")
        for i, quote in enumerate(einstein_quotes, 1):
            print(f"{i}. \"{quote['text']}\"")
        
        print("\n=== Inspirational Quotes ===")
        inspirational_quotes = scraper.filter_quotes_by_tag("inspirational")
        print(f"Found {len(inspirational_quotes)} inspirational quotes.")
        for i, quote in enumerate(inspirational_quotes[:3], 1):  # Show just first 3
            print(f"{i}. \"{quote['text']}\" by {quote['author']}")
        
        print("\nScraping completed successfully!")
        
    finally:
        # Always terminate the webdriver
        print("Terminating the webdriver...")
        scraper.terminate_webdriver()


if __name__ == "__main__":
    main() 