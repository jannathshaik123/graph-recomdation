import json
import os
from neo4j import GraphDatabase
from datetime import datetime
import time


NEO4J_URI = "bolt://localhost:7687"  
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  


base_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/yelp_training_set")
BUSINESS_FILE = os.path.join(base_path,"yelp_training_set_business.json")
REVIEW_FILE = os.path.join(base_path,"yelp_training_set_review.json")
USER_FILE = os.path.join(base_path,"yelp_training_set_user.json")
CHECKIN_FILE = os.path.join(base_path,"yelp_training_set_checkin.json")
class YelpKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def create_constraints(self):
        """Create uniqueness constraints and indexes for better performance"""
        with self.driver.session() as session:
            
            session.run("CREATE CONSTRAINT business_id IF NOT EXISTS FOR (b:Business) REQUIRE b.business_id IS UNIQUE")
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT neighborhood_name IF NOT EXISTS FOR (n:Neighborhood) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT city_name IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE")
            
            
            session.run("CREATE INDEX business_name IF NOT EXISTS FOR (b:Business) ON (b.name)")
            session.run("CREATE INDEX review_date IF NOT EXISTS FOR (r:Review) ON (r.date)")
            
            print("Constraints and indexes created.")
            
    def clear_database(self):
        """Delete all nodes and relationships from the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    def load_businesses(self, file_path, batch_size=1000):
        """Load businesses from the Yelp dataset"""
        print(f"Loading businesses from {file_path}...")
        
        count = 0
        batch = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                business = json.loads(line)
                batch.append(business)
                count += 1
                
                if len(batch) >= batch_size:
                    self._process_business_batch(batch)
                    batch = []
                    print(f"Processed {count} businesses")
                    
            
            if batch:
                self._process_business_batch(batch)
                
        print(f"Finished loading {count} businesses.")
    
    def _process_business_batch(self, batch):
        """Process a batch of business records"""
        with self.driver.session() as session:
            
            for business in batch:
                
                business_query = """
                MERGE (b:Business {business_id: $business_id})
                SET b.name = $name,
                    b.full_address = $full_address,
                    b.city = $city,
                    b.state = $state,
                    b.latitude = $latitude,
                    b.longitude = $longitude,
                    b.stars = $stars,
                    b.review_count = $review_count,
                    b.is_open = $is_open
                """
                
                session.run(
                    business_query,
                    business_id=business['business_id'],
                    name=business['name'],
                    full_address=business['full_address'],
                    city=business['city'],
                    state=business['state'],
                    latitude=business['latitude'],
                    longitude=business['longitude'],
                    stars=business['stars'],
                    review_count=business['review_count'],
                    is_open=business['open']
                )
                
                
                session.run("""
                MERGE (c:City {name: $city})
                WITH c
                MATCH (b:Business {business_id: $business_id})
                MERGE (b)-[:LOCATED_IN]->(c)
                """, city=business['city'], business_id=business['business_id'])
                
                
                for category in business['categories']:
                    session.run("""
                    MERGE (c:Category {name: $category})
                    WITH c
                    MATCH (b:Business {business_id: $business_id})
                    MERGE (b)-[:IN_CATEGORY]->(c)
                    """, category=category, business_id=business['business_id'])
                
                
                for neighborhood in business.get('neighborhoods', []):
                    if neighborhood:  
                        session.run("""
                        MERGE (n:Neighborhood {name: $neighborhood})
                        WITH n
                        MATCH (b:Business {business_id: $business_id})
                        MERGE (b)-[:IN_NEIGHBORHOOD]->(n)
                        """, neighborhood=neighborhood, business_id=business['business_id'])
    
    def load_users(self, file_path, batch_size=1000):
        """Load users from the Yelp dataset"""
        print(f"Loading users from {file_path}...")
        
        count = 0
        batch = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                user = json.loads(line)
                batch.append(user)
                count += 1
                
                if len(batch) >= batch_size:
                    self._process_user_batch(batch)
                    batch = []
                    print(f"Processed {count} users")
                    
            
            if batch:
                self._process_user_batch(batch)
                
        print(f"Finished loading {count} users.")
    
    def _process_user_batch(self, batch):
        """Process a batch of user records"""
        with self.driver.session() as session:
            for user in batch:
                
                session.run("""
                MERGE (u:User {user_id: $user_id})
                SET u.name = $name,
                    u.review_count = $review_count,
                    u.average_stars = $average_stars,
                    u.useful_votes = $useful_votes,
                    u.funny_votes = $funny_votes,
                    u.cool_votes = $cool_votes
                """, 
                user_id=user['user_id'],
                name=user['name'],
                review_count=user['review_count'],
                average_stars=user['average_stars'],
                useful_votes=user['votes']['useful'],
                funny_votes=user['votes']['funny'],
                cool_votes=user['votes']['cool']
                )
    
    def load_reviews(self, file_path, batch_size=1000):
        """Load reviews from the Yelp dataset"""
        print(f"Loading reviews from {file_path}...")
        
        count = 0
        batch = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                review = json.loads(line)
                batch.append(review)
                count += 1
                
                if len(batch) >= batch_size:
                    self._process_review_batch(batch)
                    batch = []
                    print(f"Processed {count} reviews")
                    
            
            if batch:
                self._process_review_batch(batch)
                
        print(f"Finished loading {count} reviews.")
    
    def _process_review_batch(self, batch):
        """Process a batch of review records"""
        with self.driver.session() as session:
            for review in batch:
                
                date_obj = datetime.strptime(review['date'], '%Y-%m-%d')
                
                
                
                review_id = f"{review['user_id']}_{review['business_id']}_{date_obj.strftime('%Y%m%d')}"
                
                
                session.run("""
                MATCH (u:User {user_id: $user_id})
                MATCH (b:Business {business_id: $business_id})
                CREATE (r:Review {
                    review_id: $review_id,
                    stars: $stars,
                    text: $text,
                    date: $date,
                    useful_votes: $useful_votes,
                    funny_votes: $funny_votes,
                    cool_votes: $cool_votes
                })
                CREATE (u)-[:WROTE]->(r)
                CREATE (r)-[:ABOUT]->(b)
                """,
                user_id=review['user_id'],
                business_id=review['business_id'],
                review_id=review_id,
                stars=review['stars'],
                text=review['text'],
                date=date_obj.strftime('%Y-%m-%d'),
                useful_votes=review['votes']['useful'],
                funny_votes=review['votes']['funny'],
                cool_votes=review['votes']['cool']
                )
    
    def load_checkins(self, file_path):
        """Load check-ins from the Yelp dataset"""
        print(f"Loading check-ins from {file_path}...")
        
        count = 0
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                checkin = json.loads(line)
                self._process_checkin(checkin)
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count} check-ins")
                    
        print(f"Finished loading {count} check-ins.")
    
    def _process_checkin(self, checkin):
        """Process a check-in record"""
        with self.driver.session() as session:
            
            total_checkins = sum(checkin['checkin_info'].values())
            
            session.run("""
            MATCH (b:Business {business_id: $business_id})
            MERGE (c:CheckinSummary {business_id: $business_id})
            SET c.total_checkins = $total_checkins
            MERGE (b)-[:HAS_CHECKINS]->(c)
            """,
            business_id=checkin['business_id'],
            total_checkins=total_checkins
            )
            
            
            for time_slot, count in checkin['checkin_info'].items():
                day_hour = time_slot.split('-')
                hour = int(day_hour[0])
                day = int(day_hour[1])
                
                
                day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                day_name = day_names[day]
                
                session.run("""
                MATCH (c:CheckinSummary {business_id: $business_id})
                MERGE (t:TimeSlot {hour: $hour, day: $day_name})
                MERGE (c)-[r:CHECKIN_AT]->(t)
                SET r.count = $count
                """,
                business_id=checkin['business_id'],
                hour=hour,
                day_name=day_name,
                count=count
                )


def main():
    
    kg = YelpKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        
        
        
        
        kg.create_constraints()
        
        
        start_time = time.time()
        
        
        if os.path.exists(BUSINESS_FILE):
            kg.load_businesses(BUSINESS_FILE)
        else:
            print(f"File not found: {BUSINESS_FILE}")
        
        
        if os.path.exists(USER_FILE):
            kg.load_users(USER_FILE)
        else:
            print(f"File not found: {USER_FILE}")
        
        
        if os.path.exists(REVIEW_FILE):
            kg.load_reviews(REVIEW_FILE)
        else:
            print(f"File not found: {REVIEW_FILE}")
        
        
        if os.path.exists(CHECKIN_FILE):
            kg.load_checkins(CHECKIN_FILE)
        else:
            print(f"File not found: {CHECKIN_FILE}")
        
        end_time = time.time()
        print(f"Total loading time: {end_time - start_time:.2f} seconds")
        
    finally:
        kg.close()

if __name__ == "__main__":
    main()