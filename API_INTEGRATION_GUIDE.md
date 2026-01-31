# API Integration Guide - Battery Smart NLU

## Overview
This document provides API endpoint specifications for all new intents added to the Battery Smart NLU system.

---

## Onboarding APIs

### 1. Initiate Driver Onboarding
**Intent**: `driver_onboarding`

```http
POST /api/v1/onboarding/initiate
Content-Type: application/json

{
  "phoneNumber": "9876543210",
  "driverName": "Rajesh Kumar" // optional
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

### 2. Check Onboarding Status
**Intent**: `onboarding_status`

```http
GET /api/v1/onboarding/status?driverId=DRV789012
```

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

### 3. Check Lead Creation Status
**Intent**: `lead_creation_status`

```http
GET /api/v1/onboarding/lead/status?driverId=DRV789012
```

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

### 4. Check Verification Status
**Intent**: `verification_status`

```http
GET /api/v1/onboarding/verification/status?driverId=DRV789012&verificationType=vehicle
```

**Parameters**:
- `driverId` (required): Driver ID
- `verificationType` (optional): One of `basic_info`, `location`, `address`, `vehicle`, `documents`, `all`

**Response**:
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

For `verificationType=all`:
```json
{
  "status": "success",
  "data": {
    "basic_info": {
      "status": "completed",
      "verifiedAt": "2026-01-30T11:00:00Z"
    },
    "location": {
      "status": "completed",
      "verifiedAt": "2026-01-30T12:30:00Z"
    },
    "address": {
      "status": "completed",
      "verifiedAt": "2026-01-30T13:00:00Z"
    },
    "vehicle": {
      "status": "pending",
      "expectedTime": "2026-02-01T10:00:00Z"
    },
    "documents": {
      "status": "completed",
      "verifiedAt": "2026-01-30T15:00:00Z"
    }
  }
}
```

---

## Retrofitment APIs

### 5. Schedule Retrofitment
**Intent**: `retrofitment_schedule`

```http
POST /api/v1/retrofitment/schedule
Content-Type: application/json

{
  "driverId": "DRV789012",
  "vehicleId": "DL01AB1234",
  "preferredDate": "2026-02-05",
  "preferredTime": "10:00 AM" // optional
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

### 6. Check Retrofitment Status
**Intent**: `retrofitment_status`

```http
GET /api/v1/retrofitment/status?driverId=DRV789012&vehicleId=DL01AB1234
```

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

**Status values**: `scheduled`, `in_progress`, `completed`, `cancelled`

---

## Pricing & Subscription APIs

### 7. Get Swap Price
**Intent**: `swap_price_inquiry`

```http
GET /api/v1/pricing/swap?city=Delhi&planType=basic
```

**Parameters**:
- `city` (optional): City name for location-based pricing
- `planType` (optional): Plan type (basic, standard, premium)

**Response**:
```json
{
  "status": "success",
  "data": {
    "city": "Delhi",
    "pricing": {
      "basic": {
        "pricePerSwap": 80,
        "currency": "INR"
      },
      "standard": {
        "swapsPerMonth": 30,
        "pricePerSwap": 70,
        "monthlyFee": 2100,
        "currency": "INR"
      },
      "premium": {
        "unlimitedSwaps": true,
        "monthlyFee": 3500,
        "currency": "INR"
      }
    }
  }
}
```

---

### 8. Get Subscription Plans
**Intent**: `subscription_plans`

```http
GET /api/v1/subscription/plans?city=Mumbai
```

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
        "pricePerSwap": null,
        "monthlyFee": 2200,
        "features": ["Fixed monthly fee", "30 swaps included", "Extra swaps at ₹50"],
        "recommended": true
      },
      {
        "planId": "PREMIUM_MUM",
        "planName": "Premium",
        "swapsPerMonth": "unlimited",
        "pricePerSwap": null,
        "monthlyFee": 3800,
        "features": ["Unlimited swaps", "Priority support", "No extra charges"],
        "recommended": false
      }
    ]
  }
}
```

---

### 9. Compare Plans
**Intent**: `plan_comparison`

```http
GET /api/v1/subscription/compare
```

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
      },
      {
        "feature": "Per Swap Cost",
        "basic": "₹85",
        "standard": "₹73 (effective)",
        "premium": "₹0"
      },
      {
        "feature": "Extra Swap Cost",
        "basic": "₹85",
        "standard": "₹50",
        "premium": "Free"
      },
      {
        "feature": "Priority Support",
        "basic": "No",
        "standard": "No",
        "premium": "Yes"
      }
    ],
    "recommendation": "Premium plan is best for drivers doing 50+ swaps per month"
  }
}
```

---

## DSK (Driver Seva Kendra) APIs

### 10. DSK Onboard (Before Leave)
**Intent**: `dsk_onboard`

```http
POST /api/v1/dsk/onboard
Content-Type: application/json

{
  "driverId": "DRV789012",
  "dskLocation": "DSK-DEL-042", // optional, uses nearest if not provided
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

### 11. DSK Deboard (After Leave)
**Intent**: `dsk_deboard`

```http
POST /api/v1/dsk/deboard
Content-Type: application/json

{
  "driverId": "DRV789012",
  "dskLocation": "DSK-DEL-042" // optional
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
    "dskLocation": "DSK-DEL-042",
    "accountStatus": "active",
    "deboardedAt": "2026-02-20T10:00:00Z",
    "message": "Welcome back! Your account is now active."
  }
}
```

---

### 12. Find DSK Locations
**Intent**: `dsk_location`

```http
GET /api/v1/dsk/locations?city=Delhi&location=Dwarka
```

**Parameters**:
- `city` (optional): City name
- `location` (optional): Specific area/landmark
- `latitude` (optional): For GPS-based search
- `longitude` (optional): For GPS-based search

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
        "coordinates": {
          "latitude": 28.5921,
          "longitude": 77.0460
        }
      },
      {
        "dskId": "DSK-DEL-018",
        "name": "DSK Janakpuri",
        "address": "Block C, Janakpuri, New Delhi - 110058",
        "city": "Delhi",
        "distance": 5.8,
        "distanceUnit": "km",
        "timings": "9:00 AM - 6:00 PM (Mon-Sat)",
        "contactNumber": "+91-11-87654321",
        "coordinates": {
          "latitude": 28.6217,
          "longitude": 77.0849
        }
      }
    ]
  }
}
```

---

## Driver Status APIs

### 13. Deactivate Driver
**Intent**: `driver_deactivate`

```http
POST /api/v1/driver/deactivate
Content-Type: application/json

{
  "driverId": "DRV789012",
  "centerType": "dsk", // or "ic"
  "duration": "15 days" // optional
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

### 14. Activate Driver
**Intent**: `driver_activate`

```http
POST /api/v1/driver/activate
Content-Type: application/json

{
  "driverId": "DRV789012",
  "centerType": "dsk" // or "ic"
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
    "centerName": "DSK Dwarka Sector 10",
    "message": "Welcome back! You can start taking swaps now."
  }
}
```

---

## Inactivity Center (IC) APIs

### 15. Find IC Locations
**Intent**: `ic_location`

```http
GET /api/v1/ic/locations?city=Bangalore&location=Koramangala
```

**Parameters**:
- `city` (optional): City name
- `location` (optional): Specific area/landmark
- `latitude` (optional): For GPS-based search
- `longitude` (optional): For GPS-based search

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
        "distanceUnit": "km",
        "timings": "24/7",
        "contactNumber": "+91-80-98765432",
        "batteryStorageCapacity": 50,
        "currentOccupancy": 35,
        "coordinates": {
          "latitude": 12.9352,
          "longitude": 77.6245
        }
      }
    ]
  }
}
```

---

### 16. Submit Battery at IC
**Intent**: `ic_battery_submit`

```http
POST /api/v1/ic/battery/submit
Content-Type: application/json

{
  "driverId": "DRV789012",
  "batteryId": "BAT123456",
  "icLocation": "IC-BLR-089" // optional
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
    "icLocation": "IC-BLR-089",
    "submittedAt": "2026-02-09T18:00:00Z",
    "accountStatus": "inactive",
    "receiptUrl": "https://batterymart.com/ic/receipts/IC_SUB_54321.pdf",
    "instructions": "Keep this receipt. Present it when collecting battery to reactivate account."
  }
}
```

---

## Partner Station APIs

### 17. Find Partner Stations
**Intent**: `partner_station_location`

```http
GET /api/v1/partner-stations/nearby?city=Hyderabad&location=Gachibowli
```

**Parameters**:
- `city` (optional): City name
- `location` (optional): Specific area/landmark
- `latitude` (optional): For GPS-based search
- `longitude` (optional): For GPS-based search

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
        "distanceUnit": "km",
        "availableBatteries": 12,
        "operatingHours": "6:00 AM - 11:00 PM",
        "contactNumber": "+91-40-12345678",
        "averageWaitTime": "5 minutes",
        "rating": 4.5,
        "coordinates": {
          "latitude": 17.4399,
          "longitude": 78.3489
        }
      }
    ]
  }
}
```

---

### 18. Battery Scan Help
**Intent**: `swap_battery_scan`

```http
GET /api/v1/swap/scan/help
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "instructions": [
      "Ensure QR code is clean and not damaged",
      "Hold phone steady 10-15cm from QR code",
      "Ensure good lighting on the code",
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

## Error Responses

All APIs return errors in this format:

```json
{
  "status": "error",
  "error": {
    "code": "DRIVER_NOT_FOUND",
    "message": "Driver with ID DRV789012 not found",
    "details": {
      "driverId": "DRV789012"
    }
  }
}
```

### Common Error Codes:
- `DRIVER_NOT_FOUND`: Driver ID doesn't exist
- `VEHICLE_NOT_FOUND`: Vehicle ID doesn't exist
- `BATTERY_NOT_FOUND`: Battery ID doesn't exist
- `INVALID_VERIFICATION_TYPE`: Invalid verification type provided
- `APPOINTMENT_NOT_AVAILABLE`: Requested time slot not available
- `DSK_NOT_FOUND`: DSK location doesn't exist
- `IC_NOT_FOUND`: IC location doesn't exist
- `STATION_NOT_FOUND`: Partner station doesn't exist
- `ALREADY_ONBOARDED`: Driver already onboarded at DSK
- `NOT_ONBOARDED`: Driver not onboarded at DSK (cannot deboard)
- `ACCOUNT_ALREADY_ACTIVE`: Cannot activate already active account
- `ACCOUNT_ALREADY_INACTIVE`: Cannot deactivate already inactive account

---

## Authentication

All APIs require authentication header:

```http
Authorization: Bearer <JWT_TOKEN>
X-Driver-ID: DRV789012
```

---

## Rate Limiting

- **Standard**: 100 requests per minute per driver
- **Burst**: Up to 200 requests per minute for short durations

---

## Best Practices

1. **Always include `driverId`** when available from context
2. **Use GPS coordinates** for location-based queries when available
3. **Handle optional parameters** gracefully (use defaults)
4. **Validate dates** are in future for scheduling
5. **Check battery availability** before directing to partner station
6. **Provide clear error messages** in user's language
7. **Log all deactivation/activation** events for audit

---

## Testing Endpoints

### Test Environment:
```
Base URL: https://api-test.batterymart.com
```

### Production Environment:
```
Base URL: https://api.batterymart.com
```

---

## Support

For API issues or questions:
- Email: api-support@batterymart.com
- Slack: #api-integration
- Docs: https://docs.batterymart.com/api

---

## Changelog

**v1.0 (2026-01-31)**
- Initial release with all 18 new endpoints
- Multi-language support
- Complete driver lifecycle coverage
