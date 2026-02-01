# Disease information database with treatment recommendations

DISEASE_INFO = {
    'Apple___Apple_scab': {
        'plant': 'Apple',
        'disease': 'Apple Scab',
        'description': 'A fungal disease caused by Venturia inaequalis that affects apple trees.',
        'symptoms': 'Olive-green to brown spots on leaves and fruit. Leaves may curl and fall prematurely.',
        'treatment': [
            'Remove and destroy fallen leaves',
            'Apply fungicides during spring',
            'Prune trees to improve air circulation',
            'Choose resistant varieties when planting new trees'
        ],
        'prevention': 'Plant resistant varieties and maintain good orchard sanitation.'
    },
    'Apple___Black_rot': {
        'plant': 'Apple',
        'disease': 'Black Rot',
        'description': 'A fungal disease caused by Botryosphaeria obtusa affecting leaves, fruit, and bark.',
        'symptoms': 'Frogeye leaf spots, fruit rot starting at the blossom end, and cankers on branches.',
        'treatment': [
            'Prune out dead or diseased branches',
            'Remove mummified fruits',
            'Apply fungicides during growing season',
            'Maintain tree vigor with proper fertilization'
        ],
        'prevention': 'Remove all sources of inoculum and avoid wounding trees.'
    },
    'Apple___Cedar_apple_rust': {
        'plant': 'Apple',
        'disease': 'Cedar Apple Rust',
        'description': 'A fungal disease requiring both apple and cedar/juniper trees to complete its life cycle.',
        'symptoms': 'Yellow-orange spots on leaves, sometimes with red borders. Tube-like structures on leaf undersides.',
        'treatment': [
            'Remove nearby cedar or juniper trees if possible',
            'Apply fungicides from pink bud stage through cover sprays',
            'Choose resistant apple varieties'
        ],
        'prevention': 'Plant resistant varieties and remove alternate hosts.'
    },
    'Apple___healthy': {
        'plant': 'Apple',
        'disease': 'Healthy',
        'description': 'Your apple plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care and maintenance',
            'Monitor for any changes',
            'Maintain proper watering and fertilization'
        ],
        'prevention': 'Keep up good gardening practices to maintain plant health.'
    },
    'Blueberry___healthy': {
        'plant': 'Blueberry',
        'disease': 'Healthy',
        'description': 'Your blueberry plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care and maintenance',
            'Ensure acidic soil pH (4.5-5.5)',
            'Provide adequate mulching'
        ],
        'prevention': 'Maintain proper soil conditions and regular pruning.'
    },
    'Cherry___Powdery_mildew': {
        'plant': 'Cherry',
        'disease': 'Powdery Mildew',
        'description': 'A fungal disease caused by Podosphaera clandestina affecting cherry trees.',
        'symptoms': 'White powdery coating on leaves, shoots, and sometimes fruit. Leaves may curl and distort.',
        'treatment': [
            'Apply sulfur or potassium bicarbonate fungicides',
            'Prune for better air circulation',
            'Remove and destroy infected plant parts',
            'Water at the base, not overhead'
        ],
        'prevention': 'Ensure good air circulation and avoid overhead watering.'
    },
    'Cherry___healthy': {
        'plant': 'Cherry',
        'disease': 'Healthy',
        'description': 'Your cherry plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care and maintenance',
            'Prune annually for shape and air circulation',
            'Monitor for pests and diseases'
        ],
        'prevention': 'Maintain good gardening practices.'
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'plant': 'Corn',
        'disease': 'Gray Leaf Spot',
        'description': 'A fungal disease caused by Cercospora zeae-maydis, common in humid areas.',
        'symptoms': 'Rectangular gray to tan lesions running parallel to leaf veins.',
        'treatment': [
            'Plant resistant hybrids',
            'Rotate crops to reduce inoculum',
            'Apply foliar fungicides if needed',
            'Till under crop residue after harvest'
        ],
        'prevention': 'Crop rotation and resistant varieties are key.'
    },
    'Corn___Common_rust': {
        'plant': 'Corn',
        'disease': 'Common Rust',
        'description': 'A fungal disease caused by Puccinia sorghi that produces rust-colored pustules.',
        'symptoms': 'Cinnamon-brown pustules on both leaf surfaces, may coalesce to form large areas.',
        'treatment': [
            'Plant resistant hybrids',
            'Apply fungicides if infection is severe',
            'Remove and destroy infected plant debris'
        ],
        'prevention': 'Use resistant varieties and monitor fields regularly.'
    },
    'Corn___Northern_Leaf_Blight': {
        'plant': 'Corn',
        'disease': 'Northern Leaf Blight',
        'description': 'A fungal disease caused by Exserohilum turcicum affecting corn leaves.',
        'symptoms': 'Long, elliptical gray-green to tan lesions on leaves.',
        'treatment': [
            'Plant resistant hybrids',
            'Apply foliar fungicides',
            'Rotate with non-host crops',
            'Till under infected residue'
        ],
        'prevention': 'Crop rotation and resistant hybrids are most effective.'
    },
    'Corn___healthy': {
        'plant': 'Corn',
        'disease': 'Healthy',
        'description': 'Your corn plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Ensure adequate nitrogen fertilization',
            'Monitor for pest damage'
        ],
        'prevention': 'Maintain good field practices.'
    },
    'Grape___Black_rot': {
        'plant': 'Grape',
        'disease': 'Black Rot',
        'description': 'A fungal disease caused by Guignardia bidwellii affecting grapes.',
        'symptoms': 'Brown circular lesions on leaves, mummified berries that turn black.',
        'treatment': [
            'Remove mummified fruit and infected leaves',
            'Apply fungicides from early shoot growth',
            'Prune for good air circulation',
            'Control weeds around vines'
        ],
        'prevention': 'Sanitation and timely fungicide applications.'
    },
    'Grape___Esca_(Black_Measles)': {
        'plant': 'Grape',
        'disease': 'Esca (Black Measles)',
        'description': 'A complex fungal disease affecting grapevine wood and leaves.',
        'symptoms': 'Tiger stripe pattern on leaves, dark spots on berries, vine decline.',
        'treatment': [
            'No cure exists; remove severely affected vines',
            'Protect pruning wounds',
            'Avoid stress on vines',
            'Trunk renewal may help mildly affected vines'
        ],
        'prevention': 'Protect pruning wounds and maintain vine health.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'plant': 'Grape',
        'disease': 'Leaf Blight',
        'description': 'A fungal disease causing brown spots on grape leaves.',
        'symptoms': 'Brown spots with dark borders on leaves, premature leaf drop.',
        'treatment': [
            'Apply fungicides regularly',
            'Remove infected leaves',
            'Improve air circulation through pruning',
            'Avoid overhead irrigation'
        ],
        'prevention': 'Good canopy management and fungicide program.'
    },
    'Grape___healthy': {
        'plant': 'Grape',
        'disease': 'Healthy',
        'description': 'Your grape plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Prune annually',
            'Monitor for early signs of disease'
        ],
        'prevention': 'Maintain good vineyard practices.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'plant': 'Orange',
        'disease': 'Citrus Greening (Huanglongbing)',
        'description': 'A devastating bacterial disease spread by Asian citrus psyllid.',
        'symptoms': 'Asymmetrical yellowing of leaves, lopsided bitter fruit, tree decline.',
        'treatment': [
            'No cure exists; infected trees should be removed',
            'Control Asian citrus psyllid vectors',
            'Apply nutritional sprays to extend tree life',
            'Plant certified disease-free trees'
        ],
        'prevention': 'Control psyllid populations and use certified nursery stock.'
    },
    'Peach___Bacterial_spot': {
        'plant': 'Peach',
        'disease': 'Bacterial Spot',
        'description': 'A bacterial disease caused by Xanthomonas arboricola pv. pruni.',
        'symptoms': 'Water-soaked spots on leaves that turn brown, fruit spots and cracks.',
        'treatment': [
            'Apply copper-based bactericides',
            'Prune for air circulation',
            'Remove infected plant parts',
            'Avoid overhead irrigation'
        ],
        'prevention': 'Plant resistant varieties and maintain good orchard hygiene.'
    },
    'Peach___healthy': {
        'plant': 'Peach',
        'disease': 'Healthy',
        'description': 'Your peach plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Prune annually during dormancy',
            'Monitor for pests and diseases'
        ],
        'prevention': 'Maintain good orchard practices.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'plant': 'Bell Pepper',
        'disease': 'Bacterial Spot',
        'description': 'A bacterial disease caused by Xanthomonas species affecting peppers.',
        'symptoms': 'Small, water-soaked spots on leaves, raised scabby spots on fruit.',
        'treatment': [
            'Apply copper-based bactericides',
            'Remove infected plants',
            'Avoid working with wet plants',
            'Use disease-free seeds'
        ],
        'prevention': 'Use certified seeds and practice crop rotation.'
    },
    'Pepper,_bell___healthy': {
        'plant': 'Bell Pepper',
        'disease': 'Healthy',
        'description': 'Your bell pepper plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Ensure adequate watering',
            'Monitor for pests'
        ],
        'prevention': 'Maintain good gardening practices.'
    },
    'Potato___Early_blight': {
        'plant': 'Potato',
        'disease': 'Early Blight',
        'description': 'A fungal disease caused by Alternaria solani affecting potatoes.',
        'symptoms': 'Dark brown spots with concentric rings (target spots) on lower leaves.',
        'treatment': [
            'Apply fungicides preventively',
            'Remove infected leaves',
            'Ensure adequate plant nutrition',
            'Practice crop rotation'
        ],
        'prevention': 'Crop rotation and proper plant spacing.'
    },
    'Potato___Late_blight': {
        'plant': 'Potato',
        'disease': 'Late Blight',
        'description': 'A devastating disease caused by Phytophthora infestans (caused Irish Potato Famine).',
        'symptoms': 'Water-soaked lesions on leaves, white fuzzy growth underneath, rapid plant death.',
        'treatment': [
            'Apply fungicides immediately when detected',
            'Remove and destroy infected plants',
            'Avoid overhead irrigation',
            'Harvest promptly in dry conditions'
        ],
        'prevention': 'Use certified seed potatoes and resistant varieties.'
    },
    'Potato___healthy': {
        'plant': 'Potato',
        'disease': 'Healthy',
        'description': 'Your potato plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Hill soil around stems',
            'Monitor for pest damage'
        ],
        'prevention': 'Maintain good growing conditions.'
    },
    'Raspberry___healthy': {
        'plant': 'Raspberry',
        'disease': 'Healthy',
        'description': 'Your raspberry plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Prune after fruiting',
            'Provide support for canes'
        ],
        'prevention': 'Maintain good garden practices.'
    },
    'Soybean___healthy': {
        'plant': 'Soybean',
        'disease': 'Healthy',
        'description': 'Your soybean plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Monitor for pests',
            'Maintain proper soil pH'
        ],
        'prevention': 'Rotate crops and scout fields regularly.'
    },
    'Squash___Powdery_mildew': {
        'plant': 'Squash',
        'disease': 'Powdery Mildew',
        'description': 'A common fungal disease affecting squash and related plants.',
        'symptoms': 'White powdery spots on leaves, eventually covering entire leaf surface.',
        'treatment': [
            'Apply fungicides (sulfur or potassium bicarbonate)',
            'Remove severely infected leaves',
            'Increase air circulation',
            'Water at soil level'
        ],
        'prevention': 'Choose resistant varieties and ensure good air circulation.'
    },
    'Strawberry___Leaf_scorch': {
        'plant': 'Strawberry',
        'disease': 'Leaf Scorch',
        'description': 'A fungal disease caused by Diplocarpon earlianum affecting strawberries.',
        'symptoms': 'Purple to red spots on leaves that develop tan centers, leaves may dry out.',
        'treatment': [
            'Apply fungicides in spring',
            'Remove old leaves after harvest',
            'Ensure good air circulation',
            'Avoid overhead watering'
        ],
        'prevention': 'Use certified plants and maintain bed hygiene.'
    },
    'Strawberry___healthy': {
        'plant': 'Strawberry',
        'disease': 'Healthy',
        'description': 'Your strawberry plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Remove runners as needed',
            'Mulch to prevent fruit rot'
        ],
        'prevention': 'Renovate beds annually and rotate planting areas.'
    },
    'Tomato___Bacterial_spot': {
        'plant': 'Tomato',
        'disease': 'Bacterial Spot',
        'description': 'A bacterial disease caused by Xanthomonas species affecting tomatoes.',
        'symptoms': 'Small, dark, water-soaked spots on leaves, stems, and fruit.',
        'treatment': [
            'Apply copper-based bactericides',
            'Remove infected plant parts',
            'Avoid overhead watering',
            'Use disease-free seeds and transplants'
        ],
        'prevention': 'Practice crop rotation and use certified seeds.'
    },
    'Tomato___Early_blight': {
        'plant': 'Tomato',
        'disease': 'Early Blight',
        'description': 'A fungal disease caused by Alternaria solani affecting tomatoes.',
        'symptoms': 'Dark spots with concentric rings on lower leaves, working upward.',
        'treatment': [
            'Apply fungicides preventively',
            'Remove infected lower leaves',
            'Mulch to prevent soil splash',
            'Stake or cage plants for air circulation'
        ],
        'prevention': 'Rotate crops and remove plant debris.'
    },
    'Tomato___Late_blight': {
        'plant': 'Tomato',
        'disease': 'Late Blight',
        'description': 'Same pathogen (Phytophthora infestans) that caused the Irish Potato Famine.',
        'symptoms': 'Large, irregular gray-green water-soaked spots, white mold on undersides.',
        'treatment': [
            'Apply fungicides immediately',
            'Remove and destroy infected plants',
            'Do not compost infected material',
            'Avoid wetting foliage'
        ],
        'prevention': 'Use resistant varieties and proper spacing.'
    },
    'Tomato___Leaf_Mold': {
        'plant': 'Tomato',
        'disease': 'Leaf Mold',
        'description': 'A fungal disease caused by Passalora fulva, common in greenhouses.',
        'symptoms': 'Yellow spots on upper leaf surface, olive-green mold on undersides.',
        'treatment': [
            'Improve ventilation',
            'Reduce humidity below 85%',
            'Apply fungicides if needed',
            'Remove infected leaves'
        ],
        'prevention': 'Ensure good air circulation and avoid high humidity.'
    },
    'Tomato___Septoria_leaf_spot': {
        'plant': 'Tomato',
        'disease': 'Septoria Leaf Spot',
        'description': 'A fungal disease caused by Septoria lycopersici affecting tomato leaves.',
        'symptoms': 'Small circular spots with dark borders and tan centers, tiny black dots visible.',
        'treatment': [
            'Apply fungicides when symptoms appear',
            'Remove infected lower leaves',
            'Mulch to prevent soil splash',
            'Avoid overhead watering'
        ],
        'prevention': 'Rotate crops and remove plant debris.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'plant': 'Tomato',
        'disease': 'Spider Mites',
        'description': 'Tiny arachnid pests that cause stippling and webbing on leaves.',
        'symptoms': 'Yellow stippling on leaves, fine webbing, bronzing of foliage.',
        'treatment': [
            'Spray with water to dislodge mites',
            'Apply insecticidal soap or neem oil',
            'Introduce predatory mites',
            'Avoid dusty conditions'
        ],
        'prevention': 'Keep plants well-watered and avoid drought stress.'
    },
    'Tomato___Target_Spot': {
        'plant': 'Tomato',
        'disease': 'Target Spot',
        'description': 'A fungal disease caused by Corynespora cassiicola affecting tomatoes.',
        'symptoms': 'Brown spots with concentric rings, often with yellow halos.',
        'treatment': [
            'Apply fungicides preventively',
            'Remove infected leaves',
            'Improve air circulation',
            'Avoid overhead irrigation'
        ],
        'prevention': 'Use drip irrigation and maintain plant spacing.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'plant': 'Tomato',
        'disease': 'Yellow Leaf Curl Virus',
        'description': 'A viral disease spread by whiteflies, causing severe yield loss.',
        'symptoms': 'Upward curling of leaves, yellowing, stunted growth, flower drop.',
        'treatment': [
            'No cure; remove infected plants',
            'Control whitefly populations',
            'Use reflective mulches',
            'Plant resistant varieties'
        ],
        'prevention': 'Control whiteflies and use resistant varieties.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'plant': 'Tomato',
        'disease': 'Tomato Mosaic Virus',
        'description': 'A viral disease easily spread by contact and contaminated tools.',
        'symptoms': 'Mottled light and dark green pattern on leaves, distorted growth.',
        'treatment': [
            'No cure; remove infected plants',
            'Disinfect tools with bleach solution',
            'Wash hands before handling plants',
            'Use resistant varieties'
        ],
        'prevention': 'Use virus-free seeds and sanitize equipment.'
    },
    'Tomato___healthy': {
        'plant': 'Tomato',
        'disease': 'Healthy',
        'description': 'Your tomato plant appears healthy with no visible signs of disease.',
        'symptoms': 'No disease symptoms detected.',
        'treatment': [
            'Continue regular care',
            'Stake or cage plants',
            'Water consistently at soil level'
        ],
        'prevention': 'Maintain good gardening practices and rotate crops.'
    }
}

# Class labels (must match the model's output order)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def get_disease_info(class_name):
    """Get detailed information about a disease with robust fuzzy lookup."""
    # Normalize the incoming class name
    class_name_norm = class_name.replace('(', '').replace(')', '').replace(',', '').replace('  ', ' ')
    
    # Try exact match first
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    
    # Common plant name mappings (Folder Name -> database key prefix)
    plant_aliases = {
        'bell pepper': 'pepper, bell',
        'corn maize': 'corn',
        'corn': 'corn',
        'pepper bell': 'pepper, bell'
    }
    
    target = class_name_norm.lower()
    
    # Check for plant aliasing
    for alias, replacement in plant_aliases.items():
        if target.startswith(alias + '___'):
            target = target.replace(alias, replacement, 1)
            break

    # Try case-insensitive and normalized match
    for key in DISEASE_INFO.keys():
        key_norm = key.lower().replace('(', '').replace(')', '').replace(',', '').replace('  ', ' ')
        if key_norm == target:
            return DISEASE_INFO[key]
            
    # Try underscore/space flexibility
    target_soft = target.replace('_', ' ')
    for key in DISEASE_INFO.keys():
        key_soft = key.lower().replace('(', '').replace(')', '').replace(',', '').replace('_', ' ').replace('  ', ' ')
        if key_soft == target_soft:
            return DISEASE_INFO[key]

    # Fallback default info with best effort extraction
    parts = class_name.split('___')
    plant = parts[0] if len(parts) > 0 else 'Unknown'
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    
    return {
        'plant': plant,
        'disease': disease,
        'description': f'Diagnosis: {disease} affecting {plant}.',
        'symptoms': 'Specific detail record not found, but general symptoms for this category apply.',
        'treatment': ['Consult a local agricultural extension office', 'Maintain proper soil moisture and nutrients'],
        'prevention': 'Regular monitoring and proper hygiene in the garden.'
    }
