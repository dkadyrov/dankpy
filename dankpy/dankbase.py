from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy_utils import database_exists

Base = declarative_base()

class Database(object):
    """
    
    
    """

    def __init__(self, db):
        
        self.engine = create_engine("sqlite:///{}".format(db))
        if database_exists(self.engine.url):
            Base.metadata.bind = self.engine
        else:
            Base.metadata.create_all(self.engine)
        DBSession = sessionmaker(bind=self.engine, autoflush=False)

        self.session = DBSession()

# class Item(Base): 
    #  __tablename__ = 'Item'