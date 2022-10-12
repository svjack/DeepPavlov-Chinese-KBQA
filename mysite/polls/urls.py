from django.urls import path

from . import views

urlpatterns = [
    #path('add', views.add, name='add'),
    #path('search', views.search, name='search'),
    #path('add_trans', views.add_trans, name='add_trans'),
    #path('search_trans', views.search_trans, name='search_trans'),
    #path('add_valid', views.add_valid, name='add_valid'),
    #path('get_valid', views.get_valid, name='get_valid'),
    #path('build_index', views.build_index, name='build_index'),
    #path('search_text', views.search_text, name='search_text'),
    path('extract_et', views.extract_et, name='extract_et'),
    path('property_score', views.property_score, name='property_score'),
    path('zh_entity_link', views.zh_entity_link, name='zh_entity_link'),
    path('kbqa', views.kbqa, name='kbqa'),
]
