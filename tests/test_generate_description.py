from prompts.generate_descriptions import generate_dermatology_description

if __name__ == '__main__':
    # img_path = "C:/Users/admin/PycharmProjects/Prompt2Lesion/dermo_images/Test/mel/ISIC_0034243.JPG"
    path = r"data\valid_split\mel\ISIC_0961235.jpg"
    result = generate_dermatology_description(path)
    print(result["feature_vector"])
    print(result["raw_reply"])
    print(result["usage"])