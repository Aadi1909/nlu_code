# Battery Smart - New Features Documentation

## Overview
This document outlines all the new intents, entities, and features added to the Battery Smart NLU system to support the complete driver journey from onboarding to battery swapping operations.

## New Feature Categories

### 1. Driver Onboarding Process

The onboarding process includes multiple verification steps that drivers must complete before they can start using the Battery Smart service.

#### Intents Added:
- **`driver_onboarding`**: Initiates the driver registration process
- **`onboarding_status`**: Checks the current status of onboarding
- **`lead_creation_status`**: Verifies if the driver lead has been created
- **`verification_status`**: Checks status of various verifications

#### Verification Types:
1. **Basic Info Verification**: Personal details like name, phone, email
2. **Location/Address Verification**: Residential address confirmation
3. **Vehicle Verification**: Vehicle registration and documents
4. **Document Verification**: ID proofs, license, etc.

#### Example Queries:
```
Hindi:
- "Battery Smart join करना है"
- "मेरी ऑनबोर्डिंग कहां तक हुई"
- "बेसिक इन्फो वेरिफाई हुआ क्या"

English:
- "I want to become a driver"
- "Check my onboarding status"
- "Is basic info verified"

Hinglish:
- "driver banna hai"
- "onboarding status kya hai"
- "verification kab hoga"
```

---

### 2. Vehicle Retrofitment

After successful onboarding, vehicles need retrofitment (installation of battery swap kit) to enable quick battery swapping.

#### Intents Added:
- **`retrofitment_schedule`**: Book appointment for kit installation
- **`retrofitment_status`**: Check retrofitment completion status

#### Features:
- Appointment scheduling with preferred date/time
- Real-time status tracking
- Expected completion time estimates

#### Example Queries:
```
Hindi:
- "रेट्रोफिटमेंट कब होगी"
- "किट लगाने का अपॉइंटमेंट बुक करो"
- "स्वैप किट इंस्टॉल हो गई क्या"

English:
- "Schedule kit installation"
- "When will retrofitment be done"
- "Is my vehicle swap-ready"

Hinglish:
- "kit lagane ka time batao"
- "retrofitment complete hui kya"
```

---

### 3. Swap Pricing & Subscription Plans

Comprehensive pricing information and subscription plan management.

#### Intents Added:
- **`swap_price_inquiry`**: Ask about swap costs
- **`subscription_plans`**: View all available plans
- **`plan_comparison`**: Compare different subscription tiers

#### Plan Types:
1. **Basic**: Limited swaps per month, pay-per-swap model
2. **Standard**: Fixed number of swaps per month
3. **Premium**: Unlimited swaps, best for high-usage drivers

#### Example Queries:
```
Hindi:
- "स्वैप की कीमत क्या है"
- "कौन कौन से प्लान हैं"
- "बेसिक और प्रीमियम में क्या फर्क है"

English:
- "What is swap price"
- "Show all subscription plans"
- "Compare plans"

Hinglish:
- "swap rate kya hai"
- "plan ki jankari chahiye"
- "konsa plan better hai"
```

---

### 4. DSK (Driver Seva Kendra)

Driver Seva Kendras are service centers where drivers can onboard/deboard before/after taking leave to avoid penalties.

#### Intents Added:
- **`dsk_onboard`**: Register at DSK before going on leave
- **`dsk_deboard`**: Deboard at DSK after returning from leave
- **`dsk_location`**: Find nearest DSK location

#### Purpose:
- Prevents penalties during planned leave periods
- Proper documentation of driver availability
- Ensures smooth reactivation after leave

#### Example Queries:
```
Hindi:
- "DSK पर ऑनबोर्ड करना है"
- "लीव के बाद DSK पर वापस आना है"
- "नजदीकी DSK कहां है"

English:
- "Register at Driver Seva Kendra"
- "Deboard at DSK after leave"
- "Find nearest DSK"

Hinglish:
- "chutti leni hai DSK par jana hai"
- "DSK se deboard karna hai"
- "paas mein DSK hai kya"
```

---

### 5. Driver Activation/Deactivation

Drivers can temporarily deactivate their accounts when going on leave and reactivate when ready to work.

#### Intents Added:
- **`driver_deactivate`**: Temporarily deactivate driver account
- **`driver_activate`**: Reactivate driver account

#### Deactivation Centers:
- **DSK**: For planned leaves with proper registration
- **IC**: For quick deactivation with battery submission

#### Example Queries:
```
Hindi:
- "ड्राइवर डीएक्टिवेट करना है"
- "वापस एक्टिव होना है"
- "लीव पे जाने से पहले इनएक्टिव करना है"

English:
- "Deactivate driver account"
- "Make me active again"
- "Temporarily stop driving"

Hinglish:
- "driver inactive karna hai"
- "phir se active hona hai"
```

---

### 6. IC (Inactivity Center)

Inactivity Centers allow drivers to submit their battery and deactivate their account for leave periods.

#### Intents Added:
- **`ic_location`**: Find nearest Inactivity Center
- **`ic_battery_submit`**: Submit battery at IC before leave

#### Features:
- Battery submission with receipt
- Quick deactivation without DSK registration
- Reactivation requires same IC visit with receipt

#### Example Queries:
```
Hindi:
- "इनएक्टिविटी सेंटर कहां है"
- "IC पर बैटरी जमा करनी है"
- "बैटरी सबमिट करके inactive होना है"

English:
- "Where is Inactivity Center"
- "Submit battery at IC"
- "Deposit battery and deactivate"

Hinglish:
- "IC ki location batao"
- "battery jama karni hai"
```

---

### 7. Partner Station Operations

Partner stations are third-party locations where drivers can swap batteries through a scanning-based system.

#### Intents Added:
- **`partner_station_location`**: Find partner swap stations
- **`partner_station_swap_process`**: Learn about swap process
- **`swap_battery_scan`**: Help with battery scanning issues

#### Swap Process:
1. Driver arrives at partner station
2. Gives discharged battery to partner
3. Partner scans old battery QR code on their app
4. Driver receives fully charged battery
5. Partner scans new battery QR code
6. Swap completed - saves 3-4 hours of charging time!

#### Example Queries:
```
Hindi:
- "पार्टनर स्टेशन कहां है"
- "स्वैप कैसे होता है पार्टनर के साथ"
- "बैटरी स्कैन नहीं हो रही"

English:
- "Nearest partner swap station"
- "How does partner swap work"
- "Battery not scanning"

Hinglish:
- "partner station location batao"
- "scan kaise hota hai"
```

---

### 8. Swap Benefits

Information about time and cost savings with battery swapping.

#### Intents Added:
- **`swap_time_saved`**: Compare swap time vs charging time

#### Key Benefits:
- **Time**: 2-3 minutes vs 3-4 hours
- **Earnings**: More working time = more income
- **Convenience**: No downtime waiting for charging

#### Example Queries:
```
Hindi:
- "स्वैप से कितना टाइम बचता है"
- "चार्जिंग से कितना फास्ट है"

English:
- "How much time saved with swap"
- "Swap vs charging time"

Hinglish:
- "swap se kitna time bachta hai"
```

---

## New Entities Added

### Driver Information
- **`phone_number`**: Indian mobile numbers (10 digits, starts with 6-9)
- **`driver_name`**: Driver's full name

### Verification & Status
- **`verification_type`**: Type of verification (basic_info, location, address, vehicle, documents)

### Location Identifiers
- **`dsk_location`**: DSK center ID (e.g., DSK001, DSK-DEL-042)
- **`ic_location`**: IC center ID (e.g., IC001, IC-BLR-089)
- **`partner_station_id`**: Partner station ID (e.g., PS001, PARTNER-DEL-123)

### Center Types
- **`deactivation_center`**: Where to deactivate (DSK or IC)
- **`activation_center`**: Where to activate (DSK or IC)

### Time-related
- **`leave_duration`**: How long the leave will be (e.g., "15 days", "2 weeks")
- **`preferred_date`**: Preferred appointment date
- **`preferred_time`**: Preferred appointment time
- **`leave_start_date`**: When leave starts
- **`leave_end_date`**: When leave ends

---

## Slot Configuration

All new intents have been configured with:
- **Required slots**: Must be filled before API call
- **Optional slots**: Enhances the query but not mandatory
- **Auto-fill sources**: Automatically filled from context (phone lookup, GPS, session)
- **Prompts**: Multi-language prompts to collect missing information
- **Validation**: Input validation with error messages

Example slot mapping for `dsk_onboard`:
```yaml
required_slots:
  - driver_id
optional_slots:
  - dsk_location
  - leave_start_date
  - leave_end_date
auto_fill:
  - slot: driver_id
    from: phone_lookup
```

---

## API Endpoints

All new intents are mapped to appropriate API endpoints:

### Onboarding APIs
- `POST /api/v1/onboarding/initiate`
- `GET /api/v1/onboarding/status`
- `GET /api/v1/onboarding/lead/status`
- `GET /api/v1/onboarding/verification/status`

### Retrofitment APIs
- `POST /api/v1/retrofitment/schedule`
- `GET /api/v1/retrofitment/status`

### Pricing APIs
- `GET /api/v1/pricing/swap`
- `GET /api/v1/subscription/plans`
- `GET /api/v1/subscription/compare`

### DSK APIs
- `POST /api/v1/dsk/onboard`
- `POST /api/v1/dsk/deboard`
- `GET /api/v1/dsk/locations`

### Driver Status APIs
- `POST /api/v1/driver/deactivate`
- `POST /api/v1/driver/activate`

### IC APIs
- `GET /api/v1/ic/locations`
- `POST /api/v1/ic/battery/submit`

### Partner Station APIs
- `GET /api/v1/partner-stations/nearby`
- `GET /api/v1/swap/scan/help`

---

## Response Templates

All intents have comprehensive response templates in three languages:
- **English**: For English-speaking users
- **Hindi**: For Hindi-speaking users  
- **Hinglish**: For mixed Hindi-English speakers (most common)

Response templates include:
- Success messages
- Status updates
- Error messages
- Contextual information (dates, locations, etc.)

---

## Training Data

A comprehensive training dataset has been created with:
- **Total new examples**: 100+ examples
- **Languages covered**: English, Hindi, Hinglish
- **Complexity levels**: Simple, Medium, Complex
- **Entity variations**: With and without entities

Training data file: `/data/raw/new_intents_training_data.json`

To merge with existing data, run:
```bash
python merge_training_data.py
```

---

## Testing Recommendations

### 1. Test Each Intent Category
- Onboarding flow (all 4 intents)
- Retrofitment flow (both intents)
- DSK operations (all 3 intents)
- IC operations (both intents)
- Partner station (all 3 intents)

### 2. Test Multi-language Support
- Pure Hindi queries
- Pure English queries
- Hinglish (code-mixed) queries

### 3. Test Entity Extraction
- With explicit entity mentions
- Without entities (should prompt)
- With partial entities

### 4. Test Context Handling
- Auto-fill from phone number
- Auto-fill from GPS
- Session context maintenance

---

## Migration Guide

### Step 1: Backup Current System
```bash
cp config/intents.yaml config/intents.yaml.backup
cp config/entities.yaml config/entities.yaml.backup
cp config/slots.yml config/slots.yml.backup
cp config/responses.yaml config/responses.yaml.backup
```

### Step 2: Merge Training Data
```bash
python merge_training_data.py
```

### Step 3: Retrain Models
```bash
python src/training/train_all_models.py
```

### Step 4: Test NLU Pipeline
```bash
python src/nlu/test_nlu_pipeline.py
```

### Step 5: Validate API Integrations
Ensure all new API endpoints are implemented and responding correctly.

---

## Future Enhancements

### Potential Additions:
1. **Penalty Management**: Detailed penalty queries and disputes
2. **Earnings Tracking**: Daily/weekly/monthly earning reports
3. **Battery Health Monitoring**: Predictive maintenance alerts
4. **Route Optimization**: Best routes for swaps during trips
5. **Multi-vehicle Support**: Managing multiple vehicles per driver
6. **Referral System**: Driver referral program queries
7. **Training Resources**: Access to training videos and guides
8. **Emergency Support**: Quick access to emergency helpline

---

## Contact & Support

For questions or issues with the new features:
- Check logs in `src/nlu/nlu_pipeline.py`
- Review training data quality
- Validate entity extraction accuracy
- Test API endpoint responses

---

## Version History

- **v1.0** (Current): Initial implementation of all driver lifecycle features
  - Driver onboarding (4 intents)
  - Retrofitment (2 intents)
  - Pricing & subscriptions (3 intents)
  - DSK operations (3 intents)
  - Driver status management (2 intents)
  - IC operations (2 intents)
  - Partner stations (3 intents)
  - Swap benefits (1 intent)
  
**Total New Intents**: 20
**Total New Entities**: 13
**Total Training Examples**: 100+

---

## Summary

This update provides comprehensive support for:
✅ Complete driver onboarding journey
✅ Vehicle retrofitment scheduling
✅ Transparent pricing and subscription management
✅ Leave management through DSK and IC
✅ Driver activation/deactivation
✅ Partner station operations
✅ Multi-language support (English, Hindi, Hinglish)
✅ Context-aware slot filling
✅ Comprehensive training data

The system is now capable of handling the entire driver lifecycle from initial registration to daily swap operations!
