from main import classify_and_recommend, find_iconclass_tags, recommend_images

if __name__ == "__main__":
    test_images_dir = "test_images/"
    image_name = test_images_dir + "Wyjazd_na_polowanie_z_sokolem.jpg"
    filename = "test_data.json"
    iconclass_branch = ""
    
    print("Testing classify_and_recommend function...")
    print(f"Image: {image_name}")
    print(f"Data file: {filename}")
    print()
    
    try:
        simple_recommendation, idf_recommendation, jaccard_recommendation = classify_and_recommend(
            image_name=image_name,
            iconclass_branch_to_start_from=iconclass_branch,
            filename=filename,
        )
        
        print("Simple Recommendation:")
        print(simple_recommendation)
        print()
        
        print("IDF-based Recommendation:")
        print(idf_recommendation)
        print()
        
        print("Jaccard Recommendation:")
        print(jaccard_recommendation)
        print()
        
    except Exception as e:
        print(f"Error during classification and recommendation: {e}")
        print()
    
    print("Testing find_iconclass_tags function...")
    try:
        iconclass_tags = find_iconclass_tags(
            image_name=image_name,
            iconclass_branch_to_start_from=iconclass_branch,
            search_individually="IF_NONE_FOUND"
        )
        
        print("Found Iconclass Tags:")
        for tag in iconclass_tags:
            print(iconclass_tags)
        print()
        
    except Exception as e:
        print(f"Error during iconclass tag detection: {e}")
        print()
    
    print("Testing recommend_images function...")
    try:
        test_codes = ["11A", "11B", "25F"]
        simple, idf, jaccard = recommend_images(
            filename=filename,
            iconclass_codes_list=test_codes,
            idf_impact=1
        )
        
        print("Recommendations for test codes:", test_codes)
        print("Simple:", simple)
        print("IDF-based:", idf)
        print("Jaccard:", jaccard)
        
    except Exception as e:
        print(f"Error during image recommendation: {e}")
