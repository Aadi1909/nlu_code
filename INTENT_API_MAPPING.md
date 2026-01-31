# Intent to API Mapping - Battery Smart NLU

## Complete Intent Map for Backend Development

This document maps all NLU intents to their corresponding API endpoints, required parameters, and expected responses.

---

## 🔄 Driver Onboarding & Verification

### 1. `driver_onboarding`
**Purpose**: Initiate driver registration process

**API Endpoint**: `POST /api/v1/onboarding/initiate`

**Required Parameters**:
- None (optional: `phoneNumber`, `driverName`)

**Request Body**:
```json
{
  "phoneNumber": "9876543210",
  "driverName": "Rajesh Kumar"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "leadId": "LEAD123456",
    "driverId": "DRV789012",
    "nextStep": "basic_info_verification",
    "message": "Lead created successfully"
  }
}
```

---

### 2. `onboarding_status`
**Purpose**: Check overall onboarding progress

**API Endpoint**: `GET /api/v1/onboarding/status`

**Required Parameters**:
- `driverId` (string)

**Query String**: `?driverId=DRV789012`

**Response**:
```json
{
  "status": "success",
  "data": {
    "driverId": "DRV789012",
    "currentStep": "vehicle_verification",
    "completedSteps": ["lead_creation", "basic_info_verification", "location_verification"],
    "pendingSteps": ["vehicle_verification", "retrofitment"],
    "overallProgress": 60,
    "estimatedCompletion": "2026-02-05"
  }
}
```

---

### 3. `lead_creation_status`
**Purpose**: Check if driver lead is created

**API Endpoint**: `GET /api/v1/onboarding/lead/status`

**Required Parameters**:
- `driverId` (string)

**Query String**: `?driverId=DRV789012`

**Response**:
```json
{
  "status": "success",
  "data": {
    "leadId": "LEAD123456",
    "leadStatus": "created",
    "createdAt": "2026-01-30T10:30:00Z",
    "assignedTo": "RM_Delhi_01"
  }
}
```

---

### 4. `verification_status`
**Purpose**: Check status of specific or all verifications

**API Endpoint**: `GET /api/v1/onboarding/verification/status`

**Required Parameters**:
- `driverId` (string)

**Optional Parameters**:
- `verificationType` (enum: `basic_info`, `location`, `address`, `vehicle`, `documents`, `all`)

**Query String**: `?driverId=DRV789012&verificationType=vehicle`

**Response (Single Verification)**:
```json
{
  "status": "success",
  "data": {
    "verificationType": "vehicle",
    "verificationStatus": "completed",
    "verifiedAt": "2026-01-31T14:20:00Z",
    "verifiedBy": "Verification_Agent_42",
    "remarks": "All documents verified"
  }
}
```

**Response (All Verifications)**:
```json
{
  "status": "success",
  "data": {
    "basic_info": { "status": "completed", "verifiedAt": "2026-01-30T11:00:00Z" },
    "location": { "status": "completed", "verifiedAt": "2026-01-30T12:30:00Z" },
    "address": { "status": "completed", "verifiedAt": "2026-01-30T13:00:00Z" },
    "vehicle": { "status": "pending", "expectedTime": "2026-02-01T10:00:00Z" },
    "documents": { "status": "completed", "verifiedAt": "2026-01-30T15:00:00Z" }
  }
}
```

---

## 🔧 Vehicle Retrofitment

### 5. `retrofitment_schedule`
**Purpose**: Book appointment for battery swap kit installation

**API Endpoint**: `POST /api/v1/retrofitment/schedule`

**Required Parameters**:
- `driverId` (string)
- `vehicleId` (string)

**Optional Parameters**:
- `preferredDate` (date)
- `preferredTime` (time)

**Request Body**:
```json
{
  "driverId": "DRV789012",
  "vehicleId": "DL01AB1234",
  "preferredDate": "2026-02-05",
  "preferredTime": "10:00 AM"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "appointmentId": "RETRO123456",
    "scheduledDate": "2026-02-05",
    "scheduledTime": "10:00 AM",
    "location": "Battery Smart Service Center - Dwarka",
    "address": "Sector 12, Dwarka, New Delhi - 110075",
    "estimatedDuration": "2 hours",
    "instructions": "Please bring vehicle RC and driver license"
  }
}
```

---

### 6. `retrofitment_status`
**Purpose**: Check retrofitment completion status

**API Endpoint**: `GET /api/v1/retrofitment/status`

**Required Parameters**:
- `driverId` (string)
- `vehicleId` (string)

**Query String**: `?driverId=DRV789012&vehicleId=DL01AB1234`

**Response**:
```json
{
  "status": "success",
  "data": {
    "retrofitmentStatus": "completed",
    "appointmentId": "RETRO123456",
    "completionDate": "2026-02-05T12:30:00Z",
    "kitSerialNumber": "KIT987654",
    "certificateUrl": "https://batterymart.com/certificates/KIT987654.pdf",
    "nextInspectionDue": "2026-08-05"
  }
}
```

---

## 💰 Pricing & Subscriptions

### 7. `swap_price_inquiry`
**Purpose**: Get swap pricing information

**API Endpoint**: `GET /api/v1/pricing/swap`

**Required Parameters**: None

**Optional Parameters**:
- `city` (string)
- `planType` (enum: `basic`, `standard`, `premium`)

**Query String**: `?city=Delhi&planType=basic`

**Response**:
```json
{
  "status": "success",
  "data": {
    "city": "Delhi",
    "pricing": {
      "basic": { "pricePerSwap": 80, "currency": "INR" },
      "standard": { "swapsPerMonth": 30, "pricePerSwap": 70, "monthlyFee": 2100, "currency": "INR" },
      "premium": { "unlimitedSwaps": true, "monthlyFee": 3500, "currency": "INR" }
    }
  }
}
```

---

### 8. `subscription_plans`
**Purpose**: Get all available subscription plans

**API Endpoint**: `GET /api/v1/subscription/plans`

**Required Parameters**: None

**Optional Parameters**:
- `city` (string)

**Query String**: `?city=Mumbai`

**Response**:
```json
{
  "status": "success",
  "data": {
    "city": "Mumbai",
    "plans": [
      {
        "planId": "BASIC_MUM",
        "planName": "Basic",
        "swapsPerMonth": 15,
        "pricePerSwap": 85,
        "monthlyFee": null,
        "features": ["Pay per swap", "No commitment"],
        "recommended": false
      },
      {
        "planId": "STANDARD_MUM",
        "planName": "Standard",
        "swapsPerMonth": 30,
        "monthlyFee": 2200,
        "features": ["Fixed monthly fee", "30 swaps included", "Extra swaps at ₹50"],
        "recommended": true
      },
      {
        "planId": "PREMIUM_MUM",
        "planName": "Premium",
        "swapsPerMonth": "unlimited",
        "monthlyFee": 3800,
        "features": ["Unlimited swaps", "Priority support"],
        "recommended": false
      }
    ]
  }
}
```

---

### 9. `plan_comparison`
**Purpose**: Compare subscription plans

**API Endpoint**: `GET /api/v1/subscription/compare`

**Required Parameters**: None

**Response**:
```json
{
  "status": "success",
  "data": {
    "comparison": [
      {
        "feature": "Monthly Fee",
        "basic": "Pay per use",
        "standard": "₹2200",
        "premium": "₹3800"
      },
      {
        "feature": "Swaps Included",
        "basic": "0",
        "standard": "30",
        "premium": "Unlimited"
      }
    ]
  }
}
```

---

## 🏢 DSK (Driver Seva Kendra) Operations

### 10. `dsk_onboard`
**Purpose**: Register at DSK before going on leave

**API Endpoint**: `POST /api/v1/dsk/onboard`

**Required Parameters**:
- `driverId` (string)

**Optional Parameters**:
- `dskLocation` (string)
- `leaveStartDate` (date)
- `leaveEndDate` (date)

**Request Body**:
```json
{
  "driverId": "DRV789012",
  "dskLocation": "DSK-DEL-042",
  "leaveStartDate": "2026-02-10",
  "leaveEndDate": "2026-02-20"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "onboardingId": "DSK_ONB_12345",
    "driverId": "DRV789012",
    "dskName": "DSK Dwarka Sector 10",
    "dskLocation": "DSK-DEL-042",
    "leaveStartDate": "2026-02-10",
    "leaveEndDate": "2026-02-20",
    "penaltyWaived": true,
    "onboardedAt": "2026-02-09T16:30:00Z",
    "receiptUrl": "https://batterymart.com/dsk/receipts/DSK_ONB_12345.pdf"
  }
}
```

---

### 11. `dsk_deboard`
**Purpose**: Deboard at DSK after returning from leave

**API Endpoint**: `POST /api/v1/dsk/deboard`

**Required Parameters**:
- `driverId` (string)

**Optional Parameters**:
- `dskLocation` (string)

**Request Body**:
```json
{
  "driverId": "DRV789012",
  "dskLocation": "DSK-DEL-042"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "deboardingId": "DSK_DEB_67890",
    "driverId": "DRV789012",
    "dskName": "DSK Dwarka Sector 10",
    "accountStatus": "active",
    "deboardedAt": "2026-02-20T10:00:00Z",
    "message": "Welcome back! Your account is now active."
  }
}
```

---

### 12. `dsk_location`
**Purpose**: Find nearest DSK locations

**API Endpoint**: `GET /api/v1/dsk/locations`

**Required Parameters**: None

**Optional Parameters**:
- `city` (string)
- `location` (string)
- `latitude` (float)
- `longitude` (float)

**Query String**: `?city=Delhi&location=Dwarka`

**Response**:
```json
{
  "status": "success",
  "data": {
    "dsks": [
      {
        "dskId": "DSK-DEL-042",
        "name": "DSK Dwarka Sector 10",
        "address": "Plot 15, Sector 10, Dwarka, New Delhi - 110075",
        "city": "Delhi",
        "distance": 2.5,
        "distanceUnit": "km",
        "timings": "9:00 AM - 6:00 PM (Mon-Sat)",
        "contactNumber": "+91-11-12345678",
        "coordinates": { "latitude": 28.5921, "longitude": 77.0460 }
      }
    ]
  }
}
```

---

## 👤 Driver Status Management

### 13. `driver_deactivate`
**Purpose**: Temporarily deactivate driver account

**API Endpoint**: `POST /api/v1/driver/deactivate`

**Required Parameters**:
- `driverId` (string)

**Optional Parameters**:
- `centerType` (enum: `dsk`, `ic`)
- `duration` (string)

**Request Body**:
```json
{
  "driverId": "DRV789012",
  "centerType": "dsk",
  "duration": "15 days"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "driverId": "DRV789012",
    "accountStatus": "inactive",
    "deactivatedAt": "2026-02-09T16:30:00Z",
    "centerType": "dsk",
    "centerName": "DSK Dwarka Sector 10",
    "reactivationInstructions": "Visit same DSK to reactivate",
    "noPenaltyUntil": "2026-02-24"
  }
}
```

---

### 14. `driver_activate`
**Purpose**: Reactivate driver account

**API Endpoint**: `POST /api/v1/driver/activate`

**Required Parameters**:
- `driverId` (string)

**Optional Parameters**:
- `centerType` (enum: `dsk`, `ic`)

**Request Body**:
```json
{
  "driverId": "DRV789012",
  "centerType": "dsk"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "driverId": "DRV789012",
    "accountStatus": "active",
    "activatedAt": "2026-02-20T10:00:00Z",
    "centerType": "dsk",
    "message": "Welcome back! You can start taking swaps now."
  }
}
```

---

## 🏭 Inactivity Center (IC) Operations

### 15. `ic_location`
**Purpose**: Find nearest Inactivity Centers

**API Endpoint**: `GET /api/v1/ic/locations`

**Required Parameters**: None

**Optional Parameters**:
- `city` (string)
- `location` (string)
- `latitude` (float)
- `longitude` (float)

**Query String**: `?city=Bangalore&location=Koramangala`

**Response**:
```json
{
  "status": "success",
  "data": {
    "ics": [
      {
        "icId": "IC-BLR-089",
        "name": "IC Koramangala",
        "address": "5th Block, Koramangala, Bangalore - 560095",
        "city": "Bangalore",
        "distance": 1.2,
        "timings": "24/7",
        "batteryStorageCapacity": 50,
        "currentOccupancy": 35,
        "coordinates": { "latitude": 12.9352, "longitude": 77.6245 }
      }
    ]
  }
}
```

---

### 16. `ic_battery_submit`
**Purpose**: Submit battery at IC and deactivate

**API Endpoint**: `POST /api/v1/ic/battery/submit`

**Required Parameters**:
- `driverId` (string)
- `batteryId` (string)

**Optional Parameters**:
- `icLocation` (string)

**Request Body**:
```json
{
  "driverId": "DRV789012",
  "batteryId": "BAT123456",
  "icLocation": "IC-BLR-089"
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "submissionId": "IC_SUB_54321",
    "driverId": "DRV789012",
    "batteryId": "BAT123456",
    "icName": "IC Koramangala",
    "accountStatus": "inactive",
    "submittedAt": "2026-02-09T18:00:00Z",
    "receiptUrl": "https://batterymart.com/ic/receipts/IC_SUB_54321.pdf",
    "instructions": "Keep receipt. Present when collecting battery to reactivate."
  }
}
```

---

## 🤝 Partner Station Operations

### 17. `partner_station_location`
**Purpose**: Find nearest partner swap stations

**API Endpoint**: `GET /api/v1/partner-stations/nearby`

**Required Parameters**: None

**Optional Parameters**:
- `city` (string)
- `location` (string)
- `latitude` (float)
- `longitude` (float)

**Query String**: `?city=Hyderabad&location=Gachibowli`

**Response**:
```json
{
  "status": "success",
  "data": {
    "partnerStations": [
      {
        "stationId": "PS123",
        "partnerName": "GreenCharge Station",
        "address": "Plot 42, Gachibowli, Hyderabad - 500032",
        "city": "Hyderabad",
        "distance": 0.8,
        "availableBatteries": 12,
        "operatingHours": "6:00 AM - 11:00 PM",
        "averageWaitTime": "5 minutes",
        "rating": 4.5,
        "coordinates": { "latitude": 17.4399, "longitude": 78.3489 }
      }
    ]
  }
}
```

---

### 18. `partner_station_swap_process`
**Purpose**: Get instructions for swap process

**API Endpoint**: None (returns static instructions)

**Action**: `PROVIDE_SWAP_INSTRUCTIONS`

**Response Template**:
```
Partner station swap process:
1. Reach the station
2. Give discharged battery to partner
3. Partner scans battery QR code on their app
4. You receive charged battery
5. Partner scans new battery
6. Swap complete! Saves 3-4 hours of charging time.
```

---

### 19. `swap_battery_scan`
**Purpose**: Help with battery scanning issues

**API Endpoint**: `GET /api/v1/swap/scan/help`

**Required Parameters**: None

**Response**:
```json
{
  "status": "success",
  "data": {
    "instructions": [
      "Ensure QR code is clean and not damaged",
      "Hold phone steady 10-15cm from QR code",
      "Ensure good lighting",
      "If still not scanning, contact partner or support"
    ],
    "videoUrl": "https://batterymart.com/help/battery-scan-guide.mp4",
    "supportNumber": "+91-1800-123-4567",
    "troubleshootingTips": {
      "damaged_qr": "Contact station partner for manual entry",
      "poor_light": "Use phone flashlight",
      "app_issue": "Update Battery Smart app to latest version"
    }
  }
}
```

---

### 20. `swap_time_saved`
**Purpose**: Get information about time savings

**API Endpoint**: None (returns static info)

**Action**: `PROVIDE_BENEFITS_INFO`

**Response Template**:
```
Battery swap takes just 2-3 minutes, while charging takes 3-4 hours.
You save approximately 3-4 hours per swap!
This means more earning time for you.
```

---

## 📋 Summary Table

| # | Intent | Method | Endpoint | Auth Required |
|---|--------|--------|----------|---------------|
| 1 | driver_onboarding | POST | /api/v1/onboarding/initiate | No |
| 2 | onboarding_status | GET | /api/v1/onboarding/status | Yes |
| 3 | lead_creation_status | GET | /api/v1/onboarding/lead/status | Yes |
| 4 | verification_status | GET | /api/v1/onboarding/verification/status | Yes |
| 5 | retrofitment_schedule | POST | /api/v1/retrofitment/schedule | Yes |
| 6 | retrofitment_status | GET | /api/v1/retrofitment/status | Yes |
| 7 | swap_price_inquiry | GET | /api/v1/pricing/swap | No |
| 8 | subscription_plans | GET | /api/v1/subscription/plans | No |
| 9 | plan_comparison | GET | /api/v1/subscription/compare | No |
| 10 | dsk_onboard | POST | /api/v1/dsk/onboard | Yes |
| 11 | dsk_deboard | POST | /api/v1/dsk/deboard | Yes |
| 12 | dsk_location | GET | /api/v1/dsk/locations | No |
| 13 | driver_deactivate | POST | /api/v1/driver/deactivate | Yes |
| 14 | driver_activate | POST | /api/v1/driver/activate | Yes |
| 15 | ic_location | GET | /api/v1/ic/locations | No |
| 16 | ic_battery_submit | POST | /api/v1/ic/battery/submit | Yes |
| 17 | partner_station_location | GET | /api/v1/partner-stations/nearby | No |
| 18 | swap_battery_scan | GET | /api/v1/swap/scan/help | No |
| 19 | partner_station_swap_process | - | Static Response | No |
| 20 | swap_time_saved | - | Static Response | No |

---

## 🔐 Authentication

All endpoints marked "Yes" require:
```http
Authorization: Bearer <JWT_TOKEN>
X-Driver-ID: <DRIVER_ID>
```

---

## ⚠️ Error Responses

All APIs return errors in this format:
```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": {}
  }
}
```

### Common Error Codes:
- `DRIVER_NOT_FOUND` - Driver ID doesn't exist
- `VEHICLE_NOT_FOUND` - Vehicle ID doesn't exist
- `BATTERY_NOT_FOUND` - Battery ID doesn't exist
- `INVALID_VERIFICATION_TYPE` - Invalid verification type
- `APPOINTMENT_NOT_AVAILABLE` - Time slot unavailable
- `DSK_NOT_FOUND` - DSK location doesn't exist
- `IC_NOT_FOUND` - IC location doesn't exist
- `ALREADY_ONBOARDED` - Already onboarded at DSK
- `NOT_ONBOARDED` - Not onboarded (cannot deboard)
- `ACCOUNT_ALREADY_ACTIVE` - Already active
- `ACCOUNT_ALREADY_INACTIVE` - Already inactive

---

## 📊 Rate Limiting
- **Standard**: 100 requests/minute per driver
- **Burst**: 200 requests/minute (short duration)

---

## 🌍 Multi-Language Support

All response messages should support:
- `en` - English
- `hi` - Hindi
- `hinglish` - Mixed Hindi-English

Use `Accept-Language` header:
```http
Accept-Language: hi
```

---

**Total Endpoints to Implement**: 18 REST APIs + 2 static responses = 20 intents

**Base URLs**:
- Test: `https://api-test.batterymart.com`
- Production: `https://api.batterymart.com`
